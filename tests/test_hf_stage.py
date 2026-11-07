from datetime import date
from pathlib import Path

import polars as pl

from pipeline.hf_stage import stage_for_hf


def _seed(src: Path) -> None:
    (src / "metrics").mkdir(parents=True)
    (src / "universe").mkdir()
    (src / "actions").mkdir()
    pl.DataFrame({"date": [date(2026, 10, 30), date(2026, 11, 2)], "x": [1, 2]}).write_parquet(src / "metrics/nse_2026.parquet")
    pl.DataFrame({"date": [date(2025, 12, 31)], "x": [1]}).write_parquet(src / "metrics/nse_2025.parquet")
    pl.DataFrame({"date": [date(2026, 11, 6)], "x": [1]}).write_parquet(src / "metrics/nse_latest.parquet")
    pl.DataFrame({"rebalance_date": [date(2026, 10, 1), date(2026, 11, 2)], "s": ["A", "B"]}).write_parquet(src / "universe/nse_liquid.parquet")
    pl.DataFrame({"date": [date(2026, 11, 6)]}).write_parquet(src / "actions/nse_2026.parquet")


def test_cuts_metrics_and_universe_at_month_start(tmp_path: Path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _seed(src)
    stage_for_hf(src, dst, date(2026, 11, 1))
    assert pl.read_parquet(dst / "metrics/nse_2026.parquet")["date"].to_list() == [date(2026, 10, 30)]
    assert pl.read_parquet(dst / "metrics/nse_2025.parquet").height == 1
    assert not (dst / "metrics/nse_latest.parquet").exists()
    assert pl.read_parquet(dst / "universe/nse_liquid.parquet")["s"].to_list() == ["A"]
    assert pl.read_parquet(dst / "actions/nse_2026.parquet").height == 1
    assert pl.read_parquet(src / "metrics/nse_2026.parquet").height == 2
    assert (src / "metrics/nse_latest.parquet").exists()


def test_drops_year_file_with_nothing_before_cutoff(tmp_path: Path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _seed(src)
    pl.DataFrame({"date": [date(2027, 1, 4)], "x": [1]}).write_parquet(src / "metrics/nse_2027.parquet")
    stage_for_hf(src, dst, date(2027, 1, 1))
    assert not (dst / "metrics/nse_2027.parquet").exists()
    assert pl.read_parquet(dst / "metrics/nse_2026.parquet").height == 2
