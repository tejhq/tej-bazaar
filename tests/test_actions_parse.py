import json
from datetime import date
from pathlib import Path

import polars as pl

from pipeline.actions import (
    ACTION_SCHEMA,
    CorporateAction,
    parse_actions,
    parse_bse_record,
    parse_nse_record,
    to_polars,
)
from pipeline.actions.parse import _classify

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
