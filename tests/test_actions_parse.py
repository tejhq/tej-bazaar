import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from pipeline.actions import (
    ACTION_SCHEMA,
    CorporateAction,
    parse_actions,
    parse_bse_record,
    parse_nse_record,
    to_polars,
)
from pipeline.actions.parse import _classify, _extract_details

NSE_FIXTURE = Path(__file__).parent / "fixtures" / "nse_actions_sample.json"
BSE_FIXTURE = Path(__file__).parent / "fixtures" / "bse_actions_sample.json"


def _load_nse() -> list[dict]:
    return json.loads(NSE_FIXTURE.read_text())


def _load_bse() -> list[dict]:
    return json.loads(BSE_FIXTURE.read_text())


# --- classifier ---------------------------------------------------------


def test_classify_split():
    assert _classify("Face Value Split (Sub-Division) - From Rs10/-") == "split"
    assert _classify("Stock  Split From Rs.10/- to Rs.1/-") == "split"
    # Reverse split: BSE uses "Consolidation of Shares". Same type, downstream
    # uses face_value direction to derive factor.
    assert _classify("Consolidation of Shares") == "split"
    assert _classify("Share Consolidation") == "split"


def test_classify_resolution_plan_stays_other():
    # Corporate insolvency / suspension: no price impact, not a price action.
    assert _classify("Resolution Plan -Suspension") == "other"


def test_classify_bonus():
    assert _classify("Bonus 1:1") == "bonus"
    assert _classify("Bonus issue 3:1") == "bonus"


def test_classify_rights():
    assert _classify("Rights 6:179 @ Premium Rs 1810/-") == "rights"
    assert _classify("Right Issue of Equity Shares") == "rights"


def test_classify_dividend():
    assert _classify("Interim Dividend - Rs 12 Per Share") == "dividend"
    assert _classify("Final Dividend - Rs. - 0.4400") == "dividend"
    # Distribution (InvIT) maps to dividend (same math)
    assert _classify("Income Distribution (InvIT)") == "dividend"


def test_classify_buyback():
    assert _classify("Buy Back") == "buyback"
    assert _classify("Buy Back of Shares") == "buyback"


def test_classify_demerger():
    assert _classify("Demerger") == "demerger"
    assert _classify("Spin Off") == "demerger"


def test_classify_agm():
    assert _classify("Annual General Meeting") == "agm"
    assert _classify("E.G.M.") == "agm"
    assert _classify("Extra Ordinary General Meeting") == "agm"


def test_classify_dividend_beats_agm():
    # NSE produces "Annual General Meeting/Dividend" - dividend wins (price-impacting)
    assert _classify("Annual General Meeting/Dividend") == "dividend"


def test_classify_other():
    assert _classify("Some Random Subject Text") == "other"


# --- NSE record parser --------------------------------------------------


def test_parse_nse_skips_govt_bond():
    govt = {
        "symbol": "83GS2040", "series": "GS", "subject": "Interest Payment",
        "exDate": "01-Jan-2024", "recDate": "01-Jan-2024",
        "comp": "GOVT OF INDIA", "isin": "IN0020100031",
    }
    assert parse_nse_record(govt) is None


def test_parse_nse_dividend_amount():
    rec = {
        "symbol": "HCLTECH", "series": "EQ", "isin": "INE860A01027",
        "comp": "HCL Technologies Limited", "faceVal": "2",
        "subject": "Interim Dividend - Rs 12 Per Share",
        "exDate": "19-Jan-2024", "recDate": "20-Jan-2024",
    }
    a = parse_nse_record(rec)
    assert a is not None
    assert a.exchange == "NSE"
    assert a.type == "dividend"
    assert a.cash_amount == 12.0
    assert a.symbol == "HCLTECH"
    assert a.isin == "INE860A01027"
    assert a.ex_date == date(2024, 1, 19)
    assert a.record_date == date(2024, 1, 20)


def test_parse_nse_dividend_re_prefix():
    # "Re 0.01" - sub-rupee uses "Re" not "Rs"
    rec = {
        "symbol": "AKSHAR", "series": "EQ",
        "subject": "Interim Dividend - Re 0.01 Per Share",
        "exDate": "04-Jan-2024", "recDate": "04-Jan-2024",
        "comp": "Akshar Spintex Limited", "isin": "INE256Z01017",
    }
    a = parse_nse_record(rec)
    assert a.cash_amount == 0.01


def test_parse_nse_bonus_ratio():
    rec = {
        "symbol": "ALLCARGO", "series": "EQ", "subject": "Bonus 3:1",
        "exDate": "02-Jan-2024", "recDate": "02-Jan-2024",
        "comp": "Allcargo Logistics Limited", "isin": "INE418H01029",
    }
    a = parse_nse_record(rec)
    assert a.type == "bonus"
    assert a.ratio_num == 3
    assert a.ratio_den == 1


def test_parse_nse_split_face_value():
    rec = {
        "symbol": "NESTLEIND", "series": "EQ",
        "subject": "Face Value Split (Sub-Division) - From Rs10/- Per Share To Re 1/- Per Share",
        "exDate": "05-Jan-2024", "recDate": "05-Jan-2024",
        "comp": "Nestle India Limited", "isin": "INE239A01016",
    }
    a = parse_nse_record(rec)
    assert a.type == "split"
    assert a.face_value_from == 10.0
    assert a.face_value_to == 1.0


def test_parse_nse_rights_ratio():
    rec = {
        "symbol": "GRASIM", "series": "EQ",
        "subject": "Rights 6:179 @ Premium Rs 1810/-",
        "exDate": "10-Jan-2024", "recDate": "10-Jan-2024",
        "comp": "Grasim Industries Limited", "isin": "INE047A01013",
    }
    a = parse_nse_record(rec)
    assert a.type == "rights"
    assert a.ratio_num == 6
    assert a.ratio_den == 179


def test_parse_nse_buyback_no_amount():
    rec = {
        "symbol": "DHAMPURSUG", "series": "EQ", "subject": "Buy Back",
        "exDate": "17-Jan-2024", "recDate": "17-Jan-2024",
        "comp": "Dhampur Sugar Mills Limited", "isin": "INE041A01016",
    }
    a = parse_nse_record(rec)
    assert a.type == "buyback"
    assert a.cash_amount is None
    assert a.ratio_num is None


def test_parse_nse_agm_keeps_record():
    rec = {
        "symbol": "SPICEJET", "series": "EQ", "subject": "Annual General Meeting",
        "exDate": "03-Jan-2024", "recDate": "-",
        "comp": "SPICEJET LTD", "isin": "INE285B01017",
    }
    a = parse_nse_record(rec)
    assert a.type == "agm"
    assert a.record_date is None  # "-" parses to None


def test_parse_nse_invalid_date_returns_none():
    rec = {
        "symbol": "X", "series": "EQ", "subject": "Bonus 1:1",
        "exDate": "", "recDate": "",
    }
    assert parse_nse_record(rec) is None


# --- BSE record parser --------------------------------------------------


def test_parse_bse_dividend_amount():
    rec = {
        "scrip_code": 532281, "short_name": "HCLTECH",
        "Ex_date": "19 Jan 2024", "exdate": "20240119",
        "Purpose": "Interim Dividend - Rs. - 12.0000",
        "RD_Date": "20 Jan 2024",
        "long_name": "HCL Technologies Ltd",
    }
    a = parse_bse_record(rec)
    assert a is not None
    assert a.exchange == "BSE"
    assert a.type == "dividend"
    assert a.cash_amount == 12.0
    assert a.symbol == "HCLTECH"
    assert a.isin is None  # BSE doesn't provide ISIN
    assert a.ex_date == date(2024, 1, 19)


def test_parse_bse_split_face_value():
    rec = {
        "scrip_code": 500790, "short_name": "NESTLEIND",
        "Ex_date": "05 Jan 2024", "exdate": "20240105",
        "Purpose": "Stock  Split From Rs.10/- to Rs.1/-",
        "RD_Date": "05 Jan 2024",
        "long_name": "Nestle India Ltd",
    }
    a = parse_bse_record(rec)
    assert a.type == "split"
    assert a.face_value_from == 10.0
    assert a.face_value_to == 1.0


def test_parse_bse_bonus_high_ratio():
    rec = {
        "scrip_code": 512153, "short_name": "MLINDLTD",
        "Ex_date": "05 Jan 2024", "exdate": "20240105",
        "Purpose": "Bonus issue 1:50",
        "RD_Date": "06 Jan 2024",
        "long_name": "M Lakhamsi Industries Ltd",
    }
    a = parse_bse_record(rec)
    assert a.type == "bonus"
    assert a.ratio_num == 1
    assert a.ratio_den == 50


def test_parse_bse_rights_no_ratio_in_text():
    # BSE rights records typically lack ratio info - falls through cleanly
    rec = {
        "scrip_code": 505693, "short_name": "LATIMMETAL",
        "Ex_date": "02 Jan 2024", "exdate": "20240102",
        "Purpose": "Right Issue of Equity Shares ",
        "RD_Date": "02 Jan 2024",
        "long_name": "La Tim Metal & Industries Ltd",
    }
    a = parse_bse_record(rec)
    assert a.type == "rights"
    assert a.ratio_num is None
    assert a.ratio_den is None


def test_parse_bse_invit_distribution_classified_as_dividend():
    rec = {
        "scrip_code": 542543, "short_name": "ENERGYINF",
        "Ex_date": "12 Jan 2024", "exdate": "20240112",
        "Purpose": "Income Distribution (InvIT) ",
        "RD_Date": "13 Jan 2024",
        "long_name": "Energy Infrastructure Trust",
    }
    a = parse_bse_record(rec)
    assert a.type == "dividend"


def test_parse_bse_spin_off_is_demerger():
    rec = {
        "scrip_code": 514450, "short_name": "MHLXMIRU",
        "Ex_date": "19 Apr 2024", "exdate": "20240419",
        "Purpose": "Spin Off ",
        "RD_Date": "19 Apr 2024",
        "long_name": "Mahalaxmi Rubtech Ltd",
    }
    a = parse_bse_record(rec)
    assert a.type == "demerger"


# --- batch + polars conversion -----------------------------------------


def test_parse_actions_nse_fixture_full():
    raw = _load_nse()
    actions = parse_actions(raw, "NSE")
    # All non-bond fixture rows produce a CorporateAction
    assert len(actions) > 0
    # Govt bond (series GS) was filtered
    assert all(a.exchange == "NSE" for a in actions)
    # Each captures raw_subject
    assert all(a.raw_subject for a in actions)


def test_parse_actions_bse_fixture_full():
    raw = _load_bse()
    actions = parse_actions(raw, "BSE")
    assert len(actions) > 0
    assert all(a.exchange == "BSE" for a in actions)
    # All BSE rows have isin=None
    assert all(a.isin is None for a in actions)


def test_to_polars_schema_matches():
    actions = parse_actions(_load_nse(), "NSE") + parse_actions(_load_bse(), "BSE")
    df = to_polars(actions)
    assert set(df.columns) == set(ACTION_SCHEMA.keys())
    assert df.schema["ex_date"] == pl.Date
    assert df.schema["ratio_num"] == pl.Int64
    assert df.schema["cash_amount"] == pl.Float64


def test_to_polars_empty():
    df = to_polars([])
    assert df.height == 0
    assert set(df.columns) == set(ACTION_SCHEMA.keys())


def test_corporate_action_dataclass_frozen():
    a = CorporateAction(
        exchange="NSE", symbol="X", isin="INE000A00001", company="X Ltd",
        ex_date=date(2024, 1, 1), record_date=None, type="other",
    )
    import pytest
    with pytest.raises(Exception):
        a.symbol = "Y"  # type: ignore[misc]


def test_dividend_cash_sums_same_day_payouts():
    from pipeline.actions.parse import dividend_cash

    assert dividend_cash("Dividend - Rs 28 Per Share") == 28.0
    assert dividend_cash("Interim Dividend - Rs 9 Per Share Special Dividend - Rs 18 Per Share") == 27.0
    assert dividend_cash("Final Dividend Rs.4/- Per Share And Special Dividend Rs.10/- Per Share") == 14.0
    assert dividend_cash("Interim Dividend - Rs 25 Per Sh/Dividend - Rs 65 Per Share") == 90.0
    assert dividend_cash("Annual General Meeting/Final Dividend Re 0.20/- Per Share + Special Dividend Re 0.20/- Per Share") == 0.4
    # InvIT components with no stated total: the unitholder receives the sum.
    assert dividend_cash("Distribution - Interest Payment Rs 2.72 Per Unit/ Principle Re 0.28 Per Unit") == 3.0
    assert dividend_cash("Dividend - Rs 2.83 Per Unit/Interest Amount - Rs 0.65 Per Unit/Repayment Of Spv Debt - Rs 1.85 Per Unit") == 5.33


def test_dividend_cash_total_then_breakdown_takes_total():
    from pipeline.actions.parse import dividend_cash

    assert dividend_cash("Distribution - Rs 2.230 Per Unit Comprises Of Interest - Re 0.642 Per Unit / Dividend - Rs 1.299 Per Unit") == 2.23
    assert dividend_cash("Distribution - Rs 12.71 Per Unit Consisting Of Rs 11.22 Per Unit Taxable Dividend & Rs 1.38 Per Unit Exempt") == 12.71


def test_dividend_cash_ignores_par_value_note():
    from pipeline.actions.parse import dividend_cash

    assert dividend_cash("Annual General Meeting/ Dividend Rs. 2.50/- Per Equity Share Of Rs.10/- Each") == 2.5
    assert dividend_cash("Dividend Rs 3 Per Share on face value of Rs 2") == 3.0
    assert dividend_cash("Annual General Meeting") is None


# Cash the feed writes in ways the amount pattern used to misread, all from real announcements.
CASH_SHAPES = [
    # A word ending in "r" or "re" is not a currency marker.
    ("Interim Dividend Of Re.1/- Per Share (10)% For The Financial Year 2013-2014", 1.0),
    ("Interim Dividend Rs 6/- Per Share 9purpose Revised)", 6.0),
    # ...but "Rs Per 0.09" is still an amount, and a stated total still wins.
    (
        "Dividend - Rs Per 0.09 Unit/Interest Payment - Rs 3.88 Per Unit/Return On Capital - Rs 2 Per Unit/Interest On Fixed Deposit - Rs 0.03 Per Unit",
        6.0,
    ),
    ("Interim Dividend - Rs Per 0.50 Share", 0.5),
    (
        "Distribution - Rs 3.649 Per Unit Consists Of Interest- Rs 1.8475 Per Unit/Dividend - Re 0.9003 Per Unit/ Dividend For Apr 01 To Apr 15, 2025 - Re 0.6669 /Return Of Capital Re 0.2343 Per Unit",
        3.649,
    ),
    # An amount with no marker, or a garbled one.
    ("Interim Dividend 3 Per Share", 3.0),
    ("Aannual General Meeting/Dividend - 1.40 Per Share", 1.4),
    ("Distribution - 4.8376 Per Unit", 4.8376),
    ("Annual General Meeting/Dividend Rs:- 5.50 Per Share", 5.5),
    ("Annual General Meeting/Dividend - R E 1 Per Share", 1.0),
    ("Annual General Meeting /Dividend - 1 Rs Per Share", 1.0),
    ("Interim Dividend - S 5/- Per Share (Purpose Revised)", 5.0),
    ("Annual General Meeting/Dividend 12.50/-+Special Dividend 0.50/- Per Share", 13.0),
    (
        "Distribution - 3.04316 Consisting Of Interest Rs 3.04013 Per Unit / Other Income - Rs 0.00303",
        3.04316,
    ),
    # A second payout written without a marker.
    ("Dividend - Rs 106 Per Share Special Dividend 243 Per Share", 349.0),
    ("Annual General Meeting / Dividend - Final Re 1 + Special 0.50 Per Share", 1.5),
    (
        "Dividend - Rs 2.54 Per Unit/Interest Amount - Rs 1.14 Per Unit/ Proceeds - 1.98 Per Unit",
        5.66,
    ),
    (
        "Dividend - Rs 0.06 Per Unit/Interest Amount - Rs 2.45 Per Unit/Repayment Of Spv Debt - Rs 2.56 Per Unit/Fd And Income Tax Refund- 0.03",
        5.1,
    ),
    # A mistyped decimal point.
    ("Annual General Meeting/Dividend - Re 0..30 Per Share", 0.3),
    ("Annual General Meeting/Dividend - Rs 62..50/- Per Share", 62.5),
    ("Dividend - Rs 0 .70 Per Share", 0.7),
    ("Annual General Meeting/Dividend - Rs.0125 Per Share", 0.125),
    ("Interim Dividend - Rs 09 Per Share / Special Dividend - Rs 13 Per Share", 22.0),
    # A total stated first, then broken down.
    (
        "Annual General Meeting And Dividend Rs.50/- Per Share (Special Dividend Rs.25/- Per Share And Final Dividend Rs.25/- Per Share)",
        50.0,
    ),
    ("Dividend-Rs.8.50 Per Share (Including Special Dividend Of Rs.2/- Per Share).", 8.5),
    (
        "Distribution - Re 0.45 Per Unitconsisiting Of Interest - Re 0.13 Per Unit / Return On Capital - Re 0.32 Per Unit",
        0.45,
    ),
    (
        "Distritbution - Rs 2.98 Per Unit Comprising Of Rs 1.84 Per Unit In The Form Dividend & Re 0.85 Per Unit In The Form Of Interest & Re 0.29 Per Unit In The Form Of Repayment Of Loan",
        2.98,
    ),
    # Bare numbers that are not cash stay out.
    ("Interim Dividend @ 12% On Equity Shares.", None),
    ("Agm And Final Dividend Of Rs 1.25 Per Share (12.5%)", 1.25),
    ("Distribution Rs 3 Per Unit For The First Quarter Of Fy 2018-19(Purpose Revised)", 3.0),
    ("3rd Interim Dividend Re 0.40 Per Share", 0.4),
]


@pytest.mark.parametrize("text,cash", CASH_SHAPES)
def test_dividend_cash_shapes(text, cash):
    from pipeline.actions.parse import dividend_cash

    assert dividend_cash(text) == cash


@pytest.mark.parametrize(
    "text,cash",
    [
        ("Annual General Meeting / Dividend 2024/25 - Rs 5 Per Share", 5.0),
        ("Dividend 12/07/2025 - Rs 5 Per Share", 5.0),
        ("Interim Dividend 1/2% On Equity Shares", None),
        ("Dividend Rs 1 Per Share Of Rs:- 10 Each", 1.0),
        ("Dividend Rs 1 Per Share Of R E 10 Each", 1.0),
        ("Dividend (Rs 2 Per Share And Special Dividend Rs 2 Per Share)", 4.0),
        ("Dividend Rs 0.25 Per Share (Special Dividend Rs 0.249 Per Share)", 0.499),
        (
            "Dividend (Rs 50 Per Share (Special Dividend Rs 25 And Final Dividend Rs 25))",
            50.0,
        ),
        ("Dividend Re 0.3 (Special Dividend Re 0.1 And Final Dividend Re 0.2)", 0.3),
        # A rate is not cash, and neither is a bare year closing a clause.
        ("Dividend 50 Per Cent", None),
        ("Interim Dividend 25 percent On Equity Shares", None),
        ("Final Dividend 40 Per-Cent", None),
        ("Interim Dividend 2026", None),
        ("Final Dividend 2025 / Special Dividend Rs 5 Per Share", 5.0),
        ("Interim Dividend Rs 2026", 2026.0),
    ],
)
def test_dividend_cash_guards(text, cash):
    from pipeline.actions.parse import dividend_cash

    assert dividend_cash(text) == cash


# The exchange writes a face value split a dozen ways and often without the word "from". These
# shapes all appear in the live NSE feed and all used to parse without a ratio, which silently
# became an adjustment factor of 1.0 and left prices unadjusted across the split.
SPLIT_SHAPES = [
    ("Fv Splt Frm Rs 10 To Rs 2", 10.0, 2.0),
    ("Fv Split Rs.10 To Rs.1", 10.0, 1.0),
    ("Fv Splt Frm Rs 10 To Re 1", 10.0, 1.0),
    ("Face Value Split Rs 10 To Rs 5", 10.0, 5.0),
    ("Fv Split Rs.10 To Re.1", 10.0, 1.0),
    ("Fv Split Rs.10/- To Rs.2/-", 10.0, 2.0),
    ("Fv Spl-Rs10tore1/Div-3", 10.0, 1.0),
    ("Face Value Split Rs.10/- To Rs.1/-", 10.0, 1.0),
    ("Face Value Split(Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share", 10.0, 2.0),
    ("Face Valus Split (Sub-Division) - From Rs 10/- Per To Rs 2/- Per Share", 10.0, 2.0),
    ("Bonus 1:1 / Face Value Split From 10/- To Face Value 2/-", 10.0, 2.0),
    ("Consolidation of Shares From Re 1 To Rs 10", 1.0, 10.0),
    ("Consolidation Of Equity Shares From Re 1 Per Share To Rs 10 Per Share", 1.0, 10.0),
]


@pytest.mark.parametrize("text,fv_from,fv_to", SPLIT_SHAPES)
def test_split_shapes_all_carry_a_ratio(text, fv_from, fv_to):
    assert _classify(text) == "split"
    _, _, _, got_from, got_to = _extract_details("split", text)
    assert (got_from, got_to) == (fv_from, fv_to)


def test_a_row_carrying_both_a_bonus_and_a_split_keeps_both():
    for text in ("Bonus 1:1 And Face Value Split Rs.10/- To Rs.5/-", "Bon 1:1/Fv Spl Rs.10tors.5"):
        assert _classify(text) == "split"
        num, den, _, fv_from, fv_to = _extract_details("split", text)
        assert (fv_from, fv_to) == (10.0, 5.0)
        assert (num, den) == (1, 1)


# The exchange puts a dash or a few words between "bonus" and the ratio. These rows all used to
# parse without a ratio, so the bonus got a factor of 1.0 and the adjusted series kept the drop.
BONUS_SHAPES = [
    ("Bonus - 3:1", 3, 1),
    ("Bonus- 1:2", 1, 2),
    ("Bonus Shares In The Ratio Of 1:1", 1, 1),
    ("Annual General Meeting / Dividend - Rs 3/- Per Share / Bonus - 1:2", 1, 2),
]


@pytest.mark.parametrize("text,num,den", BONUS_SHAPES)
def test_bonus_shapes_all_carry_a_ratio(text, num, den):
    assert _classify(text) == "bonus"
    assert _extract_details("bonus", text)[:2] == (num, den)


@pytest.mark.parametrize(
    "text,num,den,fv_from,fv_to",
    [
        ("Bonus - 1:1 And Face Value Split From Rs. 10 To Rs. 2", 1, 1, 10.0, 2.0),
        ("Fv Spl-Rs10tors5/Bon-2:1", 2, 1, 10.0, 5.0),
        ("Bonus 1:1 / Face Value Split From 10/- To Face Value 2/-", 1, 1, 10.0, 2.0),
    ],
)
def test_a_split_row_keeps_a_bonus_written_with_a_dash(text, num, den, fv_from, fv_to):
    assert _classify(text) == "split"
    got_num, got_den, _, got_from, got_to = _extract_details("split", text)
    assert (got_num, got_den, got_from, got_to) == (num, den, fv_from, fv_to)


def test_a_bonus_ratio_is_never_read_from_another_action():
    for text in ("Bonus And Rights 1:2", "Bonus Not Declared; Rights 1:2"):
        assert _extract_details("bonus", text)[:2] == (None, None)


# A bonus of debentures, NCRPS, preference or DVR shares hands holders a new instrument and leaves
# their equity shares alone. It must never take an equity bonus ratio. The first eight are from the
# feed; the last four are made up, with the instrument where the bonus pattern alone would read a
# ratio.
NON_EQUITY_BONUSES = [
    "Scheme Of Arrangement - Bonus Ncrps 4:1",
    "Bonus Ncrps 1:116",
    "Scheme Of Arrangement - Bonus Debentures 6:1",
    "Sch Of Agmt- Bonus Deb1:1",
    "Scheme Of Arangement- Bonus - 1 Debenture For 1 Equity Share Held",
    "Annual General Meeting/Dividend Rs.3/- Per Share/Scheme Of Arrangement - Bonus Debentures 1:1 (Purpose Revised)",
    "Bonus Preference Shares 21:1",
    "Bonus 1 Dvr : 10 Eq Share",
    "Bonus Shares In The Ratio Of 1:1 (NCRPS)",
    "Bonus - 1:10 Differential Voting Rights Shares",
    "NCRPS Bonus - 4:1",
    "Issue Of Debentures As Bonus - 6:1",
]


@pytest.mark.parametrize("text", NON_EQUITY_BONUSES)
def test_a_bonus_of_another_instrument_takes_no_equity_ratio(text):
    assert _extract_details(_classify(text), text)[:2] == (None, None)


def test_a_line_naming_another_instrument_takes_no_equity_ratio_at_all():
    # By design: no line in the feed has both, and leaving a step unadjusted is safer than applying
    # an equity factor that may belong to the debentures.
    assert _extract_details("bonus", "Bonus 1:2 / Interest On Debentures")[:2] == (None, None)


def test_dividends_mentioning_a_face_value_are_still_dividends():
    text = "Annual General Meeting / Final Dividend Rs.36.50/- Per Equity Share"
    assert _classify(text) == "dividend"
    _, _, cash, _, _ = _extract_details("dividend", text)
    assert cash == 36.50


@pytest.mark.parametrize(
    ("text", "cash"),
    [
        ("DividendRs.5/- Per Share", 5.0),
        ("Dividend OfRs.5 Per Share", 5.0),
        ("Final DividendRs 2.50", 2.5),
        ("Interim DividendRe.1/- Per Share", 1.0),
        ("Dividend Rs 2 Per ShareRs 3", 5.0),
        # A lowercase r or re closing a word is still not a marker.
        ("Interim Dividend Of Re.1/- Per Share For The Financial Year 2013-2014", 1.0),
        ("Interim Dividend Rs 6/- Per Share 9purpose Revised)", 6.0),
    ],
)
def test_dividend_cash_marker_glued_to_the_word_before(text, cash):
    from pipeline.actions.parse import dividend_cash

    assert dividend_cash(text) == cash
