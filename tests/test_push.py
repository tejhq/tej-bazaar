from datetime import date
from pathlib import Path

import polars as pl
import pytest

from pipeline.push import WriteError, partition_path, write_partitioned


def _row(d: date, symbol: str, close: float = 100.0) -> dict:
    return {
        "date": d, "symbol": symbol, "series": "EQ", "isin": f"INE{symbol[:7]:0<7}",
        "name": symbol, "open": 99.0, "high": 101.0, "low": 98.0, "close": close,
        "last": close, "prev_close": 98.5, "volume": 1000, "turnover": 99500.0, "trades": 10,
    }


_SCHEMA = {
    "date": pl.Date, "symbol": pl.Utf8, "series": pl.Utf8, "isin": pl.Utf8,
    "name": pl.Utf8, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
    "close": pl.Float64, "last": pl.Float64, "prev_close": pl.Float64,
    "volume": pl.Int64, "turnover": pl.Float64, "trades": pl.Int64,
}


def test_partition_path_layout():
    p = partition_path(Path("/tmp/out"), "NSE", date(2025, 4, 30))
    assert p == Path("/tmp/out/nse/year=2025/month=04/date=2025-04-30.parquet")


def test_partition_path_zero_pads_month():
    p = partition_path(Path("/tmp/out"), "NSE", date(2025, 1, 6))
    assert p == Path("/tmp/out/nse/year=2025/month=01/date=2025-01-06.parquet")


def test_write_single_date(tmp_path: Path):
    df = pl.DataFrame(
        [_row(date(2025, 4, 30), "RELIANCE"), _row(date(2025, 4, 30), "TCS")],
        schema=_SCHEMA,
    )
    paths = write_partitioned(df, tmp_path, "NSE")
    assert len(paths) == 1
    expected = tmp_path / "nse" / "year=2025" / "month=04" / "date=2025-04-30.parquet"
    assert paths[0] == expected
    assert expected.exists()

    rt = pl.read_parquet(expected)
    assert rt.height == 2
    assert set(rt["symbol"].to_list()) == {"RELIANCE", "TCS"}


def test_write_multi_date_splits_files(tmp_path: Path):
    df = pl.DataFrame(
        [
            _row(date(2025, 4, 29), "RELIANCE"),
            _row(date(2025, 4, 30), "RELIANCE"),
            _row(date(2025, 4, 30), "TCS"),
        ],
        schema=_SCHEMA,
    )
    paths = write_partitioned(df, tmp_path, "NSE")
    assert len(paths) == 2
    for p in paths:
        assert p.exists()
    assert pl.read_parquet(paths[0]).height + pl.read_parquet(paths[1]).height == 3


def test_write_idempotent_overwrites(tmp_path: Path):
    d = date(2025, 4, 30)
    df_v1 = pl.DataFrame([_row(d, "RELIANCE", close=1000.0)], schema=_SCHEMA)
    df_v2 = pl.DataFrame([_row(d, "RELIANCE", close=2000.0)], schema=_SCHEMA)

    write_partitioned(df_v1, tmp_path, "NSE")
    paths = write_partitioned(df_v2, tmp_path, "NSE")

    rt = pl.read_parquet(paths[0])
    assert rt.height == 1
    assert rt["close"].to_list() == [2000.0]


def test_write_empty_dataframe_is_noop(tmp_path: Path):
    df = pl.DataFrame([], schema=_SCHEMA)
    paths = write_partitioned(df, tmp_path, "NSE")
    assert paths == []
    assert not any(tmp_path.rglob("*.parquet"))


def test_write_unknown_exchange_raises(tmp_path: Path):
    df = pl.DataFrame([_row(date(2025, 4, 30), "X")], schema=_SCHEMA)
    with pytest.raises(WriteError, match="unknown exchange"):
        write_partitioned(df, tmp_path, "MCX")  # type: ignore[arg-type]


def test_write_missing_date_column_raises(tmp_path: Path):
    df = pl.DataFrame({"symbol": ["X"], "close": [100.0]})
    with pytest.raises(WriteError, match="missing 'date' column"):
        write_partitioned(df, tmp_path, "NSE")


def test_write_preserves_schema(tmp_path: Path):
    df = pl.DataFrame([_row(date(2025, 4, 30), "TCS")], schema=_SCHEMA)
    paths = write_partitioned(df, tmp_path, "NSE")
    rt = pl.read_parquet(paths[0])
    assert rt.schema["date"] == pl.Date
    assert rt.schema["volume"] == pl.Int64
    assert rt.schema["close"] == pl.Float64
