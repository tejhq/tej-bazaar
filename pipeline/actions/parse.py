"""Parse NSE/BSE corporate-action records into normalized CorporateAction rows.

Both exchanges expose a single free-text field (`subject` on NSE, `Purpose` on
BSE) that encodes type + ratio + amount. We classify the type by keyword and
extract numeric details via regex. When extraction fails (e.g. BSE rights
issues with no ratio in the text), structured fields stay None. The raw
string is preserved for downstream review.
"""

from __future__ import annotations

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

# Money amount: matches "Rs 8", "Re 0.50", "Rs.10", "Rs. - 0.0100"
_RE_MONEY = re.compile(r"(?:Re?s?\.?)\s*-?\s*(\d+(?:\.\d+)?)", re.I)

# Bonus ratio: "Bonus 3:1" or "Bonus issue 1:50"
_RE_BONUS = re.compile(r"bonus(?:\s+issue)?\s+(\d+)\s*:\s*(\d+)", re.I)

# Rights ratio: "Rights 6:179", "Right 1:10"
_RE_RIGHTS = re.compile(r"rights?\s+(\d+)\s*:\s*(\d+)", re.I)

# Split face-value transition:
# "From Rs10/- Per Share To Rs 5/- Per Share"
# "From Rs.10/- to Rs.1/-"
_RE_SPLIT_FACE = re.compile(
    r"from\s*(?:re?s?\.?\s*)?(\d+(?:\.\d+)?)\s*/?-?\s*"
    r"(?:per\s+share\s+)?to\s*(?:re?s?\.?\s*)?(\d+(?:\.\d+)?)",
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


def _classify(text: str) -> ActionType:
    """Map free-text to ActionType. Order matters: price-impacting wins over AGM."""
    t = text.lower()
    if "split" in t or "sub-division" in t or "subdivision" in t:
        return "split"
    if re.search(r"\bbonus\b", t):
        return "bonus"
    if re.search(r"\brights?\b", t):
        return "rights"
    if "dividend" in t or "distribution" in t:
        return "dividend"
    if "buy back" in t or "buyback" in t:
        return "buyback"
    if "demerger" in t or "spin off" in t or "spin-off" in t:
        return "demerger"
    if "merger" in t or "amalgamation" in t:
        return "merger"
    if "general meeting" in t or re.search(r"\b[ae]\.?g\.?m\.?\b", t):
        return "agm"
    return "other"


def _extract_details(
    action_type: ActionType, text: str
) -> tuple[int | None, int | None, float | None, float | None, float | None]:
    """Return (ratio_num, ratio_den, cash_amount, fv_from, fv_to)."""
    if action_type == "bonus":
        m = _RE_BONUS.search(text)
        if m:
            return int(m.group(1)), int(m.group(2)), None, None, None
    elif action_type == "rights":
        m = _RE_RIGHTS.search(text)
        if m:
            return int(m.group(1)), int(m.group(2)), None, None, None
    elif action_type == "split":
        m = _RE_SPLIT_FACE.search(text)
        if m:
            return None, None, None, float(m.group(1)), float(m.group(2))
    elif action_type == "dividend":
        m = _RE_MONEY.search(text)
        if m:
            return None, None, float(m.group(1)), None, None
    return None, None, None, None, None


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
