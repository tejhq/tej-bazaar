"""Parse NSE/BSE corporate-action records into normalized CorporateAction rows.

Both exchanges expose a single free-text field (`subject` on NSE, `Purpose` on
BSE) that encodes type + ratio + amount. We classify the type by keyword and
extract numeric details via regex. When extraction fails (e.g. BSE rights
issues with no ratio in the text), structured fields stay None. The raw
string is preserved for downstream review.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any

from pipeline.actions.schema import ActionType, CorporateAction

# NSE non-equity series we skip outright (govt bonds, treasury, sovereign gold).
# IV (InvIT/REIT units) is kept since they pay distributions like dividends.
_NSE_NON_EQUITY_SERIES: frozenset[str] = frozenset({
    "GS",   # Government Securities
    "GB",   # Sovereign Gold Bonds
    "TB",   # Treasury Bills
    "SG",   # State Gov Securities
})

# Money amount: matches "Rs 8", "Re 0.50", "Rs.10", "Rs. - 0.0100", "Rs:- 8/-", "R E 1" and the
# swapped "Rs Per 0.09". The marker has to start a word, or be a capital Rs or Re glued to the word
# before ("DividendRs.5", "OfRs.5"): "r 2013" in "Year 2013" and "re 9" in "Share 9purpose" are not
# money.
_RE_MONEY = re.compile(
    r"(?:\bR(?:\s?e)?s?|(?<=[a-z])(?-i:R)[es])\.?\s*:?\s*-?\s*(?:per\s+)?(\d+(?:\.\d+)?)", re.I
)

# An amount with no marker, right after a payout word: "Dividend - 1.40 Per Share", "Special
# Dividend 243 Per Share", "Proceeds - 1.98 Per Unit", "Dividend 1 Rs Per Share", "Dividend - S 5/-",
# "Div.@1.20 Per Share". It has to be followed by "/-", "per", "Rs", a breakdown or the end of a
# clause, so years ("Fy 2018-19") and percentages ("Dividend @ 12%", "Dividend 50 Per Cent") stay
# out. A bare year closing a clause ("Interim Dividend 2026") is not an amount either; "Rs 2026"
# still is.
_RE_BARE_MONEY = re.compile(
    r"\b(?:dividend|div|distribution|special|interest|proceeds|refund)\s*[:\-]?\s*(?:\.?@\s*|s\s+)?"
    r"(?!(?:19|20)\d\d\s*(?:$|[+&]|/(?!-)))"
    r"(\d+(?:\.\d+)?)(?=\s*(?:/-)?\s*(?:per\b(?!\s*-?\s*cent)|pr\b|rs\b|consist|compris|$|[+&]|/(?!\s*\d)))",
    re.I,
)

# A mistyped decimal point: "Re 0..30", "Rs 62..50/-", "Rs 0 .70".
_RE_BAD_DECIMAL = re.compile(r"(\d)\s*\.\.+\s*(\d)|(\d)\s+\.(\d)")

# Bonus ratio: "Bonus 3:1", "Bonus issue 1:50", "Bonus - 3:1", "Bonus- 1:2",
# "Bonus Shares In The Ratio Of 1:1". Only these words may come between, so another action's ratio
# ("Bonus And Rights 1:2") is never read as the bonus's.
_RE_BONUS = re.compile(
    r"\bbonus\b(?:\s+issue|\s+shares\s+in\s+the\s+ratio\s+of)?\s*-?\s*(\d+)\s*:\s*(\d+)", re.I
)

# The exchange also writes a bonus as "Bon 1:1" or "Bon-2:1" when it shares a line with a split.
_RE_BONUS_SHORT = re.compile(r"\bbon\s*-?\s*(\d+)\s*:\s*(\d+)", re.I)

# An instrument other than equity: "Bonus Ncrps 4:1", "Bonus Debentures 6:1", "Bonus Deb1:1",
# "Bonus Preference Shares 21:1", "Bonus 1 Dvr : 10 Eq Share". Holders of such a bonus get the
# instrument and keep their equity shares. A line naming one takes no equity bonus ratio, even if an
# equity bonus could share the line: a wrong equity factor rescales years of history, a missing one
# leaves the step visible as before.
_RE_NON_EQUITY = re.compile(
    r"\b(?:(?:debentures?|ncds?|bonds?|ncrps|ccps|ocps|rps|preference|pref|dvrs?|warrants?)\b"
    r"|deb(?=\d)|differential\s+voting)",
    re.I,
)

# Rights ratio: "Rights 6:179", "Right 1:10"
_RE_RIGHTS = re.compile(r"rights?\s+(\d+)\s*:\s*(\d+)", re.I)

# Split face-value transition. The exchange writes this a dozen ways and "from" is often absent:
#   "From Rs10/- Per Share To Rs 5/- Per Share"   "From Rs.10/- to Rs.1/-"
#   "Face Value Split Rs 10 To Re 1"              "Fv Splt Frm Rs 10 To Rs 2"
#   "Fv Spl-Rs10tore1/Div-3"                      "Bon 1:1/Fv Spl Rs.10tors.5"
#   "From Rs 10/- Per To Rs 2/- Per Share"        "From 10/- To Face Value 2/-"
# The old pattern required a literal "from" and left 44 rows without a ratio, which silently became
# a factor of 1.0 and left prices unadjusted across the split. A face value needs a currency marker
# unless "from" or "face value" introduces it.
_RE_SPLIT_FACE = re.compile(
    r"(?:(?:from|frm)\s*(?:re?s?\.?)?|re?s?\.?)\s*(\d+(?:\.\d+)?)\s*(?:/-)?\s*"
    r"(?:per\b\s*(?:(?:share|unit)s?\s*)?)?(?:each\s*)?"
    r"to\s*(?:face\s+value\s*(?:re?s?\.?)?|re?s?\.?)\s*(\d+(?:\.\d+)?)",
    re.I,
)


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    s = s.strip()
    if not s or s == "-":
        return None
    for fmt in ("%d-%b-%Y", "%d %b %Y", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


# "Dividend" abbreviated or misspelt: "Int Div-Rs.2/-", "Agm/Div-Rs.3.50", "Fin.Div.Re.0.60",
# "Interim Dividned", "Divdend", "Divided", "Dividen", "Diviednd", "Ividend", and "Distributn". A word
# starting "div" with a later "d" is a misspelt dividend; "division" has none, and "sub-divided" is a
# split.
_RE_DIVIDEND_WORD = re.compile(r"\bdiv\b|(?<!sub-)\bdiv\w*d|\bividend|\bdistributn")

# InvIT and REIT payouts that never say "dividend": "Interest Amount - Rs 1.20 Per Unit/ Return On
# Capital - Rs 1.40 Per Unit", "Interest And Principal Repayment - Rs 5.60 Per Unit". They count only
# with an amount; "Interest Payment" alone has nothing to apply.
_RE_UNIT_PAYOUT = re.compile(r"nterest|return o[nf] capital|principal|repayment")

# An AGM line whose meeting words are followed straight by a per-share amount is its dividend:
# "Annual General Meeting- Re 0.85 Per Share".
_RE_AGM_AMOUNT = re.compile(
    r"(?:general\s+meeting|\bagm)\W*\bre?s?\.?\s*-?\s*\d+(?:\.\d+)?\s*(?:/-)?\s*per\s+(?:equity\s+)?share"
)


def _classify(text: str) -> ActionType:
    """Map free-text to ActionType. Order matters: price-impacting wins over AGM."""
    t = text.lower()
    # "Consolidation of Shares" is a reverse split; same type, inverted face values.
    if (
        "split" in t
        # The exchange abbreviates: "Fv Splt Frm Rs 10 To Rs 2", "Fv Spl-Rs10tore1/Div-3".
        or "splt" in t
        or re.search(r"\bfv\s*spl", t)
        or "sub-division" in t
        or "subdivision" in t
        or re.search(r"consolidation of (?:equity )?shares", t)
        or "share consolidation" in t
    ):
        return "split"
    if re.search(r"\bbonus\b", t):
        return "bonus"
    if re.search(r"\brights?\b", t):
        return "rights"
    if "dividend" in t or "distribution" in t or _RE_DIVIDEND_WORD.search(t):
        return "dividend"
    if "buy back" in t or "buyback" in t:
        return "buyback"
    if "demerger" in t or "spin off" in t or "spin-off" in t:
        return "demerger"
    if "merger" in t or "amalgamation" in t:
        return "merger"
    if (
        _RE_UNIT_PAYOUT.search(t)
        and not re.search(r"\b(?:reduction|scheme)\b", t)
        and dividend_cash(text) is not None
    ):
        return "dividend"
    if "general meeting" in t or re.search(r"\b[ae]\.?g\.?m\.?\b", t):
        return "dividend" if _RE_AGM_AMOUNT.search(t) else "agm"
    return "other"


def _bonus_ratio(text: str) -> tuple[int | None, int | None]:
    """Equity bonus ratio, or none when the line names another instrument (`_RE_NON_EQUITY`)."""
    if _RE_NON_EQUITY.search(text):
        return None, None
    m = _RE_BONUS.search(text) or _RE_BONUS_SHORT.search(text)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _extract_details(
    action_type: ActionType, text: str
) -> tuple[int | None, int | None, float | None, float | None, float | None]:
    """Return (ratio_num, ratio_den, cash_amount, fv_from, fv_to)."""
    if action_type == "bonus":
        num, den = _bonus_ratio(text)
        if num is not None:
            return num, den, None, None, None
    elif action_type == "rights":
        m = _RE_RIGHTS.search(text)
        if m:
            return int(m.group(1)), int(m.group(2)), None, None, None
    elif action_type == "split":
        m = _RE_SPLIT_FACE.search(text)
        if m:
            # Some rows carry both in one line: "Bonus 1:1 And Face Value Split Rs.10/- To Rs.5/-".
            # Keep the bonus ratio too so the factor can apply both.
            num, den = _bonus_ratio(text)
            return num, den, None, float(m.group(1)), float(m.group(2))
    elif action_type == "dividend":
        cash = dividend_cash(text)
        if cash is not None:
            return None, None, cash, None, None
    return None, None, None, None, None


# "Per Equity Share Of Rs.10/- Each", "Per Equity Share Re.1/- Each", "face value Rs 2": a par value,
# not cash.
_RE_PAR_NOTE = re.compile(
    r"(?:(?:share|unit)s?\s+of\s+|face\s+value\s+(?:of\s+)?)(?:R(?:\s?e)?s?\.?)\s*:?\s*-?\s*\d+(?:\.\d+)?\s*(?:/-)?\s*(?:each)?"
    r"|(?<=\d)(?:/-)?\s+per\s+(?:equity\s+)?(?:share|unit)s?\s+R(?:\s?e)?s?\.?\s*:?\s*-?\s*\d+(?:\.\d+)?\s*(?:/-)?\s*each",
    re.I,
)
_RE_TOTAL_THEN_BREAKDOWN = re.compile(r"compris|consis|includ|break", re.I)


def _amount(digits: str) -> float:
    # A leading zero before two or more digits is a lost decimal point: "Rs.0125" is 0.125, the same
    # dividend the company paid in the five years before. Two digits ("Rs 09") are a padded integer.
    if re.fullmatch(r"0\d{2,}", digits):
        return float("0." + digits[1:])
    return float(digits)


def _bracket_depth(text: str, pos: int) -> int:
    return text.count("(", 0, pos) - text.count(")", 0, pos)


def dividend_cash(text: str) -> float | None:
    """Cash per share going ex on the date, from the announcement text.

    One amount is the common case. Several amounts mean several payouts on
    the same ex date ("Final Dividend Rs 4 And Special Dividend Rs 10",
    "Interim Dividend Rs 25 / Dividend Rs 65", InvIT interest plus return
    of capital plus dividend): the price drops by the total, so they are
    summed. The exception is a line that states its total and then breaks
    it down ("Distribution Rs 2.23 Per Unit Comprises Of ...", "Dividend
    Rs 8.50 (Including Special Dividend Of Rs 2)", or a bracket whose
    amounts add up to the first), where the first amount already is the
    total. A par value note ("Per Share Of Rs 10 Each") is stripped before
    extraction.
    """
    cleaned = _RE_BAD_DECIMAL.sub(lambda m: f"{m[1] or m[3]}.{m[2] or m[4]}", text or "")
    cleaned = _RE_PAR_NOTE.sub(" ", cleaned)
    found = sorted(
        [(m.start(), m[1]) for m in _RE_MONEY.finditer(cleaned)]
        + [(m.start(1), m[1]) for m in _RE_BARE_MONEY.finditer(cleaned)]
    )
    amounts = [_amount(digits) for _, digits in found]
    if not amounts:
        return None
    if len(amounts) == 1:
        return amounts[0]
    if _RE_TOTAL_THEN_BREAKDOWN.search(cleaned):
        return amounts[0]
    depth = _bracket_depth(cleaned, found[0][0])
    if all(_bracket_depth(cleaned, pos) > depth for pos, _ in found[1:]) and math.isclose(
        sum(amounts[1:]), amounts[0]
    ):
        return amounts[0]
    return round(sum(amounts), 4)


def parse_nse_record(rec: dict[str, Any]) -> CorporateAction | None:
    """Parse one NSE corporate-action JSON record. Returns None for non-equity."""
    series = (rec.get("series") or "").strip().upper()
    if series in _NSE_NON_EQUITY_SERIES:
        return None

    subject = (rec.get("subject") or "").strip()
    ex = _parse_date(rec.get("exDate"))
    if ex is None:
        return None

    action_type = _classify(subject)
    rn, rd, cash, fv_from, fv_to = _extract_details(action_type, subject)

    return CorporateAction(
        exchange="NSE",
        symbol=(rec.get("symbol") or "").strip(),
        isin=(rec.get("isin") or "").strip() or None,
        company=(rec.get("comp") or "").strip(),
        ex_date=ex,
        record_date=_parse_date(rec.get("recDate")),
        type=action_type,
        ratio_num=rn,
        ratio_den=rd,
        cash_amount=cash,
        face_value_from=fv_from,
        face_value_to=fv_to,
        raw_subject=subject,
    )


def parse_bse_record(
    rec: dict[str, Any],
    scrip_to_isin: dict[str, str] | None = None,
) -> CorporateAction | None:
    """Parse one BSE corporate-action JSON record.

    BSE's corp-action feed lacks ISIN. When `scrip_to_isin` is supplied
    (built from the BSE active-equities master), ISIN is joined via
    `rec["scrip_code"]`. Without the map, isin stays None.
    """
    purpose = (rec.get("Purpose") or "").strip()
    ex = _parse_date(rec.get("exdate")) or _parse_date(rec.get("Ex_date"))
    if ex is None:
        return None

    action_type = _classify(purpose)
    rn, rd, cash, fv_from, fv_to = _extract_details(action_type, purpose)

    isin: str | None = None
    if scrip_to_isin is not None:
        sc = rec.get("scrip_code")
        if sc is not None:
            isin = scrip_to_isin.get(str(sc).strip())

    return CorporateAction(
        exchange="BSE",
        symbol=(rec.get("short_name") or "").strip(),
        isin=isin,
        company=(rec.get("long_name") or "").strip(),
        ex_date=ex,
        record_date=_parse_date(rec.get("RD_Date")),
        type=action_type,
        ratio_num=rn,
        ratio_den=rd,
        cash_amount=cash,
        face_value_from=fv_from,
        face_value_to=fv_to,
        raw_subject=purpose,
    )


def parse_actions(
    records: list[dict[str, Any]],
    exchange: str,
    scrip_to_isin: dict[str, str] | None = None,
) -> list[CorporateAction]:
    if exchange.upper() == "NSE":
        return [a for a in (parse_nse_record(r) for r in records) if a is not None]
    return [
        a for a in (parse_bse_record(r, scrip_to_isin) for r in records)
        if a is not None
    ]
