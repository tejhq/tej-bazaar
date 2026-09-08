"""tej-bazaar CLI: ingest, adjust, derive, publish NSE/BSE EOD data.

Commands:
    tej-bazaar fetch DATE [--exchange NSE|BSE|both]
    tej-bazaar backfill --from D --to D [--exchange NSE|BSE|both]
    tej-bazaar info [--data-dir PATH]
    tej-bazaar actions fetch (--year YYYY | --from D --to D) [--exchange NSE|BSE|both]
    tej-bazaar actions adjust (--year YYYY | --all-years) [--exchange NSE|BSE|both]
    tej-bazaar symbol-history build [--exchange NSE|BSE|both]
    tej-bazaar metrics build (--year YYYY | --all-years) [--exchange NSE|BSE|both]
    tej-bazaar reconcile --from D --to D (-s SYM1,SYM2 | --top N) [--exchange NSE|BSE]
    tej-bazaar publish [--repo REPO] [--data-dir PATH] [--dry-run]
    tej-bazaar version
"""

from __future__ import annotations

import time
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Iterable, Optional

import polars as pl
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from pipeline import __version__, holidays
from pipeline.actions import (
    ActionsFetchError,
    back_adjust,
    compute_action_factors,
    fetch_bse_actions,
    fetch_nse_actions,
    load_bse_scrip_to_isin,
    parse_actions,
    resolve_isin_via_symbol_history,
    to_polars as actions_to_polars,
)
from pipeline.fetch import (
    BhavcopyFetchError,
    BhavcopyNotFoundError,
    fetch_bse,
    fetch_nse,
)
from pipeline.parse import parse_bhavcopy
from pipeline.publish import DEFAULT_REPO_ID, PublishError, publish_to_hf
from pipeline.publish_r2 import DEFAULT_BUCKET as DEFAULT_R2_BUCKET
from pipeline.publish_r2 import PublishError as PublishR2Error
from pipeline.publish_r2 import prune_r2_prefix, publish_to_r2, pull_from_r2
from pipeline.push import partition_path, write_partitioned
from pipeline.reconcile import (
    YahooFetchError,
    fetch_yahoo_adjusted,
    reconcile_symbol,
    summarize,
)
from pipeline.metrics import compute_returns, compute_rolling
from pipeline.symbol_history import build_symbol_history
from pipeline.transform import transform

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUT_DIR = Path("data/out")
DEFAULT_ACTIONS_CACHE_DIR = Path("data/raw/actions")
DEFAULT_ACTIONS_OUT_DIR = Path("data/out/actions")
DEFAULT_PRICES_ADJUSTED_DIR = Path("data/out/prices_adjusted")
DEFAULT_SYMBOL_HISTORY_DIR = Path("data/out/symbol_history")
DEFAULT_METRICS_DIR = Path("data/out/metrics")
DEFAULT_UNIVERSE_DIR = Path("data/out/universe")


class ExchangeChoice(str, Enum):
    NSE = "NSE"
    BSE = "BSE"
    BOTH = "both"


def _read_bhavcopy_mixed(paths: Iterable[Path | str]) -> pl.DataFrame:
    """Read bhavcopy parquet that may mix daily files and year rollups.

    Rollups carry `year`/`month` as REAL columns (written by `compact` so
    rollup-only queries can still filter on them) while daily files keep
    them only in the hive path. Feeding that mixed set to one
    `pl.read_parquet([...])` or a plain `pl.concat` trips polars'
    uniform-schema check (SchemaError / ShapeError: width 16 vs 14).
    Normalise per-file: drop year/month, then concat relaxed.
    """
    frames = []
    for p in paths:
        f = pl.read_parquet(p, hive_partitioning=False)
        for col in ("year", "month"):
            if col in f.columns:
                f = f.drop(col)
        frames.append(f)
    return pl.concat(frames, how="vertical_relaxed")


def _isin_anchors_for(prices_root: Path) -> pl.DataFrame:
    """Symbol to ISIN map across the legacy cutover, from all years on disk.

    Only (date, symbol, isin) are read, so this is cheap even over 16 years.
    """
    from pipeline.isin_backfill import isin_anchors

    files = sorted(prices_root.rglob("*.parquet"))
    if not files:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "isin": pl.Utf8})
    frames = [
        pl.read_parquet(f, columns=["date", "symbol", "isin"], hive_partitioning=False)
        for f in files
    ]
    return isin_anchors(pl.concat(frames, how="vertical_relaxed"))


def _price_years_on_disk(prices_root: Path) -> list[int]:
    """Years present under a per-exchange price tree, rollups included.

    Globbing `year=*/month=*/*.parquet` misses years that exist only as a
    compacted rollup (`year=YYYY/<ex>_YYYY.parquet`), which after the
    daily cron's prune step is EVERY year. Walk all parquet and parse the
    `year=` path segment instead.
    """
    return sorted({
        int(seg.split("=", 1)[1])
        for p in prices_root.rglob("*.parquet")
        for seg in p.parts
        if seg.startswith("year=")
    })


BANNER = r"""
████████╗███████╗     ██╗  ██████╗  █████╗ ███████╗ █████╗  █████╗ ██████╗
╚══██╔══╝██╔════╝     ██║  ██╔══██╗██╔══██╗╚══███╔╝██╔══██╗██╔══██╗██╔══██╗
   ██║   █████╗       ██║  ██████╔╝███████║  ███╔╝ ███████║███████║██████╔╝
   ██║   ██╔══╝  ██   ██║  ██╔══██╗██╔══██║ ███╔╝  ██╔══██║██╔══██║██╔══██╗
   ██║   ███████╗╚█████╔╝  ██████╔╝██║  ██║███████╗██║  ██║██║  ██║██║  ██║
   ╚═╝   ╚══════╝ ╚════╝   ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝
"""

console = Console()
app = typer.Typer(
    name="tej-bazaar",
    help="Free, open EOD market data for India, NSE and BSE.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


def _banner() -> None:
    console.print(Text(BANNER, style="bold cyan"))
    console.print(
        Panel.fit(
            f"[bold]tej-bazaar[/bold] [dim]v{__version__}[/dim]\n"
            "[dim]EOD bhavcopy → parquet pipeline[/dim]",
            border_style="cyan",
        )
    )


def _parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError as e:
        raise typer.BadParameter(f"date must be YYYY-MM-DD ({e})") from e


def _exchanges(choice: ExchangeChoice) -> list[str]:
    if choice == ExchangeChoice.BOTH:
        return ["NSE", "BSE"]
    return [choice.value]


def _fetch_one(exchange: str, d: date, raw_dir: Path) -> Path:
    sub = raw_dir / exchange.lower()
    if exchange == "NSE":
        return fetch_nse(d, sub)
    return fetch_bse(d, sub)


def _run_one(
    exchange: str,
    d: date,
    raw_dir: Path,
    out_dir: Path,
    progress: Progress,
    task_id,
) -> tuple[Path, int] | None:
    """Run pipeline for a single (exchange, date). Returns (path, row_count) or None."""
    tag = f"[cyan]{exchange}[/cyan]"
    progress.update(task_id, description=f"{tag} fetch     {d}")
    csv_path = _fetch_one(exchange, d, raw_dir)

    progress.update(task_id, description=f"{tag} parse     {d}")
    df = parse_bhavcopy(csv_path)

    progress.update(task_id, description=f"{tag} transform {d}")
    df = transform(df, exchange=exchange)  # type: ignore[arg-type]

    progress.update(task_id, description=f"{tag} write     {d}")
    paths = write_partitioned(df, out_dir, exchange)  # type: ignore[arg-type]
    return (paths[0], df.height) if paths else None


@app.command()
def fetch(
    date_str: Annotated[
        str,
        typer.Argument(help="Trading date in YYYY-MM-DD format", metavar="DATE"),
    ],
    exchange: Annotated[
        ExchangeChoice,
        typer.Option("--exchange", "-e", help="Exchange to fetch", case_sensitive=False),
    ] = ExchangeChoice.NSE,
    raw_dir: Annotated[
        Path, typer.Option("--raw-dir", help="Directory for downloaded CSVs")
    ] = DEFAULT_RAW_DIR,
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Directory for output parquet")
    ] = DEFAULT_OUT_DIR,
) -> None:
    """Run the full pipeline for a single trading [bold]DATE[/bold]."""
    _banner()
    d = _parse_date(date_str)
    exchanges = _exchanges(exchange)

    results: list[tuple[str, Path, int]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=30),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("starting", total=len(exchanges))
        for ex in exchanges:
            try:
                result = _run_one(ex, d, raw_dir, out_dir, progress, task)
            except BhavcopyNotFoundError as e:
                console.print(f"[yellow]skip[/yellow] {ex} {d}: {e}")
                progress.advance(task)
                continue
            except BhavcopyFetchError as e:
                console.print(f"[red]error[/red] {ex} {d}: {e}")
                raise typer.Exit(code=1) from e
            progress.advance(task)
            if result is None:
                console.print(f"[yellow]{ex} {d}: no rows after transform[/yellow]")
                continue
            path, rows = result
            results.append((ex, path, rows))

    if not results:
        return
    body = "\n".join(
        f"[green]✔[/green] [bold]{ex}[/bold]  {rows} rows  [dim]→ {path}[/dim]"
        for ex, path, rows in results
    )
    console.print(Panel.fit(body, border_style="green"))


@app.command()
def backfill(
    from_date: Annotated[str, typer.Option("--from", help="Start date YYYY-MM-DD")],
    to_date: Annotated[str, typer.Option("--to", help="End date YYYY-MM-DD (inclusive)")],
    exchange: Annotated[
        ExchangeChoice,
        typer.Option("--exchange", "-e", help="Exchange to backfill", case_sensitive=False),
    ] = ExchangeChoice.NSE,
    raw_dir: Annotated[Path, typer.Option("--raw-dir")] = DEFAULT_RAW_DIR,
    out_dir: Annotated[Path, typer.Option("--out-dir")] = DEFAULT_OUT_DIR,
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing/--overwrite",
            help="Skip dates whose parquet already exists",
        ),
    ] = True,
) -> None:
    """Run pipeline over a date range. Non-trading days are skipped."""
    _banner()
    start = _parse_date(from_date)
    end = _parse_date(to_date)
    if end < start:
        raise typer.BadParameter("--to must be on or after --from")

    exchanges = _exchanges(exchange)
    sessions = holidays.trading_days_between(start, end, "NSE")  # NSE+BSE share calendar
    console.print(
        f"[bold]{len(sessions)}[/bold] trading days × {len(exchanges)} exchange(s) "
        f"in [cyan]{start}[/cyan] → [cyan]{end}[/cyan]"
    )

    counts = {ex: {"ok": 0, "skipped": 0, "failed": 0} for ex in exchanges}
    total = len(sessions) * len(exchanges)
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("backfill", total=total)
        for d in sessions:
            for ex in exchanges:
                existing = partition_path(out_dir, ex, d)  # type: ignore[arg-type]
                if skip_existing and existing.exists():
                    counts[ex]["skipped"] += 1
                    progress.update(task, advance=1, description=f"[dim]skip[/dim]    {ex} {d}")
                    continue
                try:
                    _run_one(ex, d, raw_dir, out_dir, progress, task)
                    counts[ex]["ok"] += 1
                except BhavcopyNotFoundError:
                    counts[ex]["skipped"] += 1
                except BhavcopyFetchError as e:
                    counts[ex]["failed"] += 1
                    console.print(f"  [red]fail[/red] {ex} {d}: {e}")
                except Exception as e:  # noqa: BLE001  # keep loop alive on any per-day error
                    counts[ex]["failed"] += 1
                    console.print(f"  [red]error[/red] {ex} {d}: {type(e).__name__}: {e}")
                progress.update(task, advance=1)

    summary = Table(title="backfill summary", border_style="cyan")
    summary.add_column("Exchange", style="bold")
    summary.add_column("Written", justify="right", style="green")
    summary.add_column("Skipped", justify="right", style="yellow")
    summary.add_column("Failed", justify="right", style="red")
    for ex in exchanges:
        c = counts[ex]
        summary.add_row(ex, str(c["ok"]), str(c["skipped"]), str(c["failed"]))
    console.print(summary)


@app.command()
def info(
    data_dir: Annotated[Path, typer.Option("--data-dir")] = DEFAULT_OUT_DIR,
) -> None:
    """Show what data is currently on disk."""
    _banner()
    if not data_dir.exists():
        console.print(f"[yellow]no data directory at[/yellow] {data_dir}")
        return

    table = Table(title="Local parquet inventory", border_style="cyan")
    table.add_column("Exchange", style="bold")
    table.add_column("Files", justify="right")
    table.add_column("Earliest")
    table.add_column("Latest")
    table.add_column("Total size", justify="right")

    for ex_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        files = sorted(ex_dir.rglob("*.parquet"))
        if not files:
            continue
        dates = [_date_from_path(p) for p in files]
        size_mb = sum(p.stat().st_size for p in files) / 1024 / 1024
        table.add_row(
            ex_dir.name.upper(),
            str(len(files)),
            min(dates).isoformat(),
            max(dates).isoformat(),
            f"{size_mb:.2f} MB",
        )
    console.print(table)


@app.command()
def publish(
    data_dir: Annotated[
        Path, typer.Option("--data-dir", help="Local parquet root to push")
    ] = DEFAULT_OUT_DIR,
    repo: Annotated[
        str, typer.Option("--repo", help="HuggingFace dataset repo id")
    ] = DEFAULT_REPO_ID,
    message: Annotated[
        str | None,
        typer.Option("-m", "--message", help="Commit message"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="List files, do not upload"),
    ] = False,
    delete_pattern: Annotated[
        list[str] | None,
        typer.Option(
            "--delete-pattern",
            help="Remote glob to delete when absent from this upload "
            "(repeatable, e.g. `nse/year=*/month=*`)",
        ),
    ] = None,
    card: Annotated[
        Path | None,
        typer.Option("--card", help="Dataset card to upload as the repo README.md"),
    ] = None,
) -> None:
    """Push partitioned parquet under DATA_DIR to a HuggingFace dataset repo."""
    _banner()
    try:
        result = publish_to_hf(
            data_dir,
            repo_id=repo,
            commit_message=message,
            dry_run=dry_run,
            delete_patterns=list(delete_pattern) if delete_pattern else None,
            card=card,
        )
    except PublishError as e:
        console.print(f"[red]publish failed[/red]: {e}")
        raise typer.Exit(code=1) from e

    mb = result.total_bytes / 1024 / 1024
    body = (
        f"[bold]repo[/bold]   {result.repo_id}\n"
        f"[bold]files[/bold]  {result.file_count}\n"
        f"[bold]size[/bold]   {mb:.2f} MB"
    )
    if dry_run:
        body += "\n[yellow]dry-run, nothing uploaded[/yellow]"
    elif result.commit_url:
        body += f"\n[dim]commit: {result.commit_url}[/dim]"
    console.print(Panel.fit(body, border_style="green" if not dry_run else "yellow"))


actions_app = typer.Typer(
    name="actions",
    help="Corporate-action ingestion (dividends, splits, bonus, rights, ...).",
    no_args_is_help=True,
)
app.add_typer(actions_app)


@actions_app.command("fetch")
def actions_fetch(
    from_date: Annotated[
        str | None,
        typer.Option("--from", help="Start date YYYY-MM-DD (or use --year)"),
    ] = None,
    to_date: Annotated[
        str | None,
        typer.Option("--to", help="End date YYYY-MM-DD inclusive (or use --year)"),
    ] = None,
    year: Annotated[
        int | None,
        typer.Option(
            "--year",
            help="Fetch full calendar year (Jan 1 to Dec 31). "
            "Writes annual parquet `<exchange>_<year>.parquet` for stable cron output.",
        ),
    ] = None,
    exchange: Annotated[
        ExchangeChoice,
        typer.Option("--exchange", "-e", help="Exchange to fetch", case_sensitive=False),
    ] = ExchangeChoice.BOTH,
    cache_dir: Annotated[
        Path, typer.Option("--cache-dir", help="Directory for raw JSON cache")
    ] = DEFAULT_ACTIONS_CACHE_DIR,
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Directory for normalized parquet output")
    ] = DEFAULT_ACTIONS_OUT_DIR,
    refresh_scrip_map: Annotated[
        bool,
        typer.Option(
            "--refresh-scrip-map",
            help="Re-pull BSE scrip-to-ISIN master (otherwise uses cached copy)",
        ),
    ] = False,
) -> None:
    """Fetch corporate actions for a date range and write normalized parquet."""
    _banner()
    if year is not None:
        if from_date or to_date:
            raise typer.BadParameter("--year is mutually exclusive with --from/--to")
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        stem_for = lambda ex: f"{ex.lower()}_{year}"  # noqa: E731
    else:
        if not from_date or not to_date:
            raise typer.BadParameter("provide --from and --to (or --year)")
        start = _parse_date(from_date)
        end = _parse_date(to_date)
        stem_for = lambda ex: f"{ex.lower()}_{start:%Y%m%d}_{end:%Y%m%d}"  # noqa: E731
    if end < start:
        raise typer.BadParameter("--to must be on or after --from")

    exchanges = _exchanges(exchange)
    out_dir.mkdir(parents=True, exist_ok=True)

    bse_scrip_map: dict[str, str] | None = None
    if "BSE" in exchanges:
        try:
            bse_scrip_map = load_bse_scrip_to_isin(cache_dir, refresh=refresh_scrip_map)
        except ActionsFetchError as e:
            console.print(f"[yellow]warn[/yellow] BSE scrip map unavailable, ISIN will be null: {e}")
            bse_scrip_map = None

    summary = Table(title="actions fetch", border_style="cyan")
    summary.add_column("Exchange", style="bold")
    summary.add_column("Raw rows", justify="right")
    summary.add_column("Parsed", justify="right", style="green")
    summary.add_column("ISIN match", justify="right")
    summary.add_column("Output", style="dim")

    for ex in exchanges:
        try:
            if ex == "NSE":
                raw = fetch_nse_actions(start, end, cache_dir)
            else:
                raw = fetch_bse_actions(start, end, cache_dir)
        except ActionsFetchError as e:
            console.print(f"[red]error[/red] {ex}: {e}")
            raise typer.Exit(code=1) from e

        actions = parse_actions(raw, ex, scrip_to_isin=bse_scrip_map if ex == "BSE" else None)
        df = actions_to_polars(actions)
        with_isin = sum(1 for a in actions if a.isin)
        out_path = out_dir / f"{stem_for(ex)}.parquet"
        df.write_parquet(out_path)
        summary.add_row(ex, str(len(raw)), str(df.height),
                        f"{with_isin}/{df.height}", str(out_path))

    console.print(summary)


@actions_app.command("adjust")
def actions_adjust(
    year: Annotated[
        Optional[int],
        typer.Option("--year", help="Calendar year to adjust (omit with --all-years)"),
    ] = None,
    all_years: Annotated[
        bool,
        typer.Option("--all-years", help="Re-adjust every year of bhavcopy on disk"),
    ] = False,
    exchange: Annotated[
        ExchangeChoice,
        typer.Option("--exchange", "-e", help="Exchange to adjust", case_sensitive=False),
    ] = ExchangeChoice.BOTH,
    prices_dir: Annotated[
        Path,
        typer.Option("--prices-dir", help="Root of partitioned bhavcopy parquet"),
    ] = DEFAULT_OUT_DIR,
    actions_dir: Annotated[
        Path,
        typer.Option("--actions-dir", help="Directory containing actions parquet"),
    ] = DEFAULT_ACTIONS_OUT_DIR,
    out_dir: Annotated[
        Path,
        typer.Option("--out-dir", help="Directory for adjusted prices parquet"),
    ] = DEFAULT_PRICES_ADJUSTED_DIR,
) -> None:
    """Compute back-adjusted prices using corporate actions.

    Reads bhavcopy parquet under `<prices-dir>/<ex>/year=<year>/**` and the
    actions parquet at `<actions-dir>/<ex>_<year>.parquet`. Writes adjusted
    prices to `<out-dir>/<ex>_<year>.parquet`.

    Use `--all-years` to re-adjust every year on disk. Required when a new
    corporate action lands that affects older years' adjusted closes (a 2026
    bonus retroactively changes the 2024 adj_close for that ISIN).
    """
    _banner()
    if all_years and year is not None:
        raise typer.BadParameter("--year and --all-years are mutually exclusive")
    if not all_years and year is None:
        raise typer.BadParameter("provide --year YYYY or --all-years")

    exchanges = _exchanges(exchange)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = Table(title="actions adjust", border_style="cyan")
    summary.add_column("Exchange", style="bold")
    summary.add_column("Year", justify="right")
    summary.add_column("Price rows", justify="right")
    summary.add_column("Actions", justify="right")
    summary.add_column("Adjusted rows", justify="right", style="green")
    summary.add_column("Output", style="dim")

    for ex in exchanges:
        if all_years:
            years = _price_years_on_disk(prices_dir / ex.lower())
            if not years:
                console.print(f"[yellow]skip[/yellow] {ex}: no bhavcopy parquet under "
                              f"{prices_dir / ex.lower()}")
                continue
        else:
            years = [year]

        anchors = _isin_anchors_for(prices_dir / ex.lower())
        if not anchors.is_empty():
            console.print(f"[dim]{ex}: backfilling ISIN for {anchors.height} symbols "
                          f"continuous across the legacy cutover[/dim]")
        for y in years:
            _adjust_one_year(ex, y, prices_dir, actions_dir, out_dir, summary, anchors)

    console.print(summary)


def _adjust_one_year(
    ex: str,
    year: int,
    prices_dir: Path,
    actions_dir: Path,
    out_dir: Path,
    summary: Table,
    anchors: pl.DataFrame | None = None,
) -> None:
    prices_glob = list((prices_dir / ex.lower()).rglob(f"year={year}/**/*.parquet"))
    if not prices_glob:
        console.print(f"[yellow]skip[/yellow] {ex} {year}: no bhavcopy parquet under "
                      f"{prices_dir / ex.lower()}/year={year}")
        return

    prices = _read_bhavcopy_mixed(prices_glob)
    if anchors is not None and not anchors.is_empty():
        from pipeline.isin_backfill import apply_isin_anchors

        prices = apply_isin_anchors(prices, anchors)

    # Back-adjusting prices in `year` must apply EVERY corporate action
    # whose ex_date > any price date in this slice, i.e. all future
    # actions across later annual files. Otherwise a 2024 price won't
    # see a 2025 1:1 bonus and ends up ~2x the true adjusted close.
    future_action_paths = sorted(
        actions_dir.glob(f"{ex.lower()}_*.parquet")
    )
    future_action_paths = [
        p for p in future_action_paths
        if _year_from_actions_filename(p) is not None
        and _year_from_actions_filename(p) >= year
    ]
    if not future_action_paths:
        console.print(f"[yellow]skip[/yellow] {ex} {year}: no actions parquet for years >= {year}")
        return
    actions = pl.concat([pl.read_parquet(p) for p in future_action_paths])

    # NSE sometimes tags actions to a stale ISIN (e.g. HDFC merger
    # legacy). Re-resolve each action's ISIN against what its symbol
    # actually traded under on ex_date in our price history. The
    # symbol-history lookup needs to span every action's ex_date, so
    # build it from ALL years of prices, not just the year being
    # adjusted.
    all_prices_paths = sorted(
        (prices_dir / ex.lower()).rglob("*.parquet")
    )
    if all_prices_paths:
        all_prices = pl.concat(
            [pl.read_parquet(p, columns=["symbol", "isin", "date"])
             for p in all_prices_paths]
        )
        actions = resolve_isin_via_symbol_history(actions, all_prices)

    factors = compute_action_factors(actions, prices)
    adjusted = back_adjust(prices, factors)
    out_path = out_dir / f"{ex.lower()}_{year}.parquet"
    adjusted.write_parquet(out_path)
    summary.add_row(ex, str(year), str(prices.height), str(actions.height),
                    str(adjusted.height), str(out_path))


symbol_history_app = typer.Typer(
    name="symbol-history",
    help="Build per-ISIN symbol-history intervals from bhavcopy series.",
    no_args_is_help=True,
)
app.add_typer(symbol_history_app)


@symbol_history_app.command("build")
def symbol_history_build(
    exchange: Annotated[
        ExchangeChoice,
        typer.Option("--exchange", "-e", help="Exchange to build", case_sensitive=False),
    ] = ExchangeChoice.BOTH,
    prices_dir: Annotated[
        Path,
        typer.Option("--prices-dir", help="Root of partitioned bhavcopy parquet"),
    ] = DEFAULT_OUT_DIR,
    out_dir: Annotated[
        Path,
        typer.Option("--out-dir", help="Directory for symbol-history parquet"),
    ] = DEFAULT_SYMBOL_HISTORY_DIR,
) -> None:
    """Scan all bhavcopy years for an exchange, emit symbol-history parquet.

    Output is a single file per exchange (`<out-dir>/<ex>.parquet`) covering
    the full price history on disk: one row per (isin, contiguous symbol run).
    """
    _banner()
    exchanges = _exchanges(exchange)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = Table(title="symbol-history build", border_style="cyan")
    summary.add_column("Exchange", style="bold")
    summary.add_column("Price files", justify="right")
    summary.add_column("Price rows", justify="right")
    summary.add_column("Intervals", justify="right", style="green")
    summary.add_column("Output", style="dim")

    for ex in exchanges:
        files = sorted((prices_dir / ex.lower()).rglob("*.parquet"))
        if not files:
            console.print(f"[yellow]skip[/yellow] {ex}: no bhavcopy parquet under "
                          f"{prices_dir / ex.lower()}")
            continue

        prices = _read_bhavcopy_mixed(files)
        from pipeline.isin_backfill import apply_isin_anchors, isin_anchors

        prices = apply_isin_anchors(prices, isin_anchors(prices))
        history = build_symbol_history(prices, ex)
        out_path = out_dir / f"{ex.lower()}.parquet"
        history.write_parquet(out_path)
        summary.add_row(ex, str(len(files)), str(prices.height),
                        str(history.height), str(out_path))

    console.print(summary)


metrics_app = typer.Typer(
    name="metrics",
    help="Build derived metrics (returns + rolling) from back-adjusted prices.",
    no_args_is_help=True,
)
app.add_typer(metrics_app)


@metrics_app.command("build")
def metrics_build(
    year: Annotated[
        Optional[int],
        typer.Option("--year", help="Calendar year to build (omit with --all-years)"),
    ] = None,
    all_years: Annotated[
        bool,
        typer.Option("--all-years", help="Build metrics for every adjusted year on disk"),
    ] = False,
    exchange: Annotated[
        ExchangeChoice,
        typer.Option("--exchange", "-e", help="Exchange to build", case_sensitive=False),
    ] = ExchangeChoice.BOTH,
    adjusted_dir: Annotated[
        Path,
        typer.Option("--adjusted-dir", help="Directory containing per-year adjusted parquet"),
    ] = DEFAULT_PRICES_ADJUSTED_DIR,
    out_dir: Annotated[
        Path,
        typer.Option("--out-dir", help="Directory for per-year metrics parquet"),
    ] = DEFAULT_METRICS_DIR,
) -> None:
    """Build per-year metrics parquet (returns + rolling windows).

    Output is `<out-dir>/<ex>_<year>.parquet`, one row per (isin, date)
    in `<year>`. Always reads ALL prior years for the same exchange
    even when building a single year, because rolling windows (up to
    252 trading days) need a full lookback to populate.
    """
    _banner()
    if all_years and year is not None:
        raise typer.BadParameter("--year and --all-years are mutually exclusive")
    if not all_years and year is None:
        raise typer.BadParameter("provide --year YYYY or --all-years")

    exchanges = _exchanges(exchange)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = Table(title="metrics build", border_style="cyan")
    summary.add_column("Exchange", style="bold")
    summary.add_column("Year", justify="right")
    summary.add_column("Rows", justify="right", style="green")
    summary.add_column("Output", style="dim")

    for ex in exchanges:
        files = sorted(adjusted_dir.glob(f"{ex.lower()}_*.parquet"))
        files = [
            p for p in files
            if _year_from_actions_filename(p) is not None
        ]
        if not files:
            console.print(f"[yellow]skip[/yellow] {ex}: no adjusted parquet under {adjusted_dir}")
            continue

        years_on_disk = sorted({_year_from_actions_filename(p) for p in files})
        if all_years:
            target_years = years_on_disk
        else:
            if year not in years_on_disk:
                console.print(
                    f"[yellow]skip[/yellow] {ex} {year}: no adjusted parquet for that year"
                )
                continue
            target_years = [year]

        # Read every available year to seed the 252-day rolling window;
        # filter to target on write.
        full_adjusted = pl.concat(
            [pl.read_parquet(p) for p in files]
        ).select(["isin", "date", "symbol", "adj_close", "volume", "turnover"])

        returns = compute_returns(full_adjusted)
        rolling = compute_rolling(full_adjusted).drop(["symbol", "adj_close"])
        metrics = returns.join(rolling, on=["isin", "date"], how="left")

        for y in target_years:
            slice_ = metrics.filter(pl.col("date").dt.year() == y)
            out_path = out_dir / f"{ex.lower()}_{y}.parquet"
            slice_.write_parquet(out_path)
            summary.add_row(ex, str(y), str(slice_.height), str(out_path))

    console.print(summary)


@app.command()
def reconcile(
    from_date: Annotated[str, typer.Option("--from", help="Start date YYYY-MM-DD")],
    to_date: Annotated[str, typer.Option("--to", help="End date YYYY-MM-DD inclusive")],
    symbols: Annotated[
        str | None,
        typer.Option(
            "--symbols",
            "-s",
            help="Comma-separated symbol list (e.g. NESTLEIND,RELIANCE). Use --top N for auto-pick.",
        ),
    ] = None,
    top: Annotated[
        int | None,
        typer.Option(
            "--top",
            "-n",
            help="Auto-pick top N symbols by mean daily turnover over the date range.",
        ),
    ] = None,
    exchange: Annotated[
        ExchangeChoice,
        typer.Option("--exchange", "-e", help="Exchange (NSE or BSE, not both)", case_sensitive=False),
    ] = ExchangeChoice.NSE,
    adjusted_dir: Annotated[
        Path,
        typer.Option("--adjusted-dir", help="Directory containing per-year adjusted parquet"),
    ] = DEFAULT_PRICES_ADJUSTED_DIR,
    tolerance_pct: Annotated[
        float,
        typer.Option("--tolerance", help="Per-row diff tolerance, percent"),
    ] = 0.5,
    sleep_between: Annotated[
        float,
        typer.Option("--sleep", help="Seconds to wait between Yahoo calls (rate-limit politeness)"),
    ] = 1.5,
) -> None:
    """Compare local adjusted closes against Yahoo's adjusted closes.

    Loads every per-year adjusted parquet that overlaps the requested range,
    filters to the requested symbols, fetches Yahoo's adjclose series for
    each, and reports per-symbol + overall match rate within tolerance.
    """
    _banner()
    if exchange == ExchangeChoice.BOTH:
        raise typer.BadParameter("reconcile takes one exchange at a time")
    ex = exchange.value
    start = _parse_date(from_date)
    end = _parse_date(to_date)
    if end < start:
        raise typer.BadParameter("--to must be on or after --from")

    if (symbols is None) == (top is None):
        raise typer.BadParameter("provide exactly one of --symbols or --top")

    years = list(range(start.year, end.year + 1))
    parquet_paths = [adjusted_dir / f"{ex.lower()}_{y}.parquet" for y in years]
    existing = [p for p in parquet_paths if p.exists()]
    if not existing:
        console.print(f"[red]error[/red] no adjusted parquet found for years {years} in {adjusted_dir}")
        raise typer.Exit(code=1)

    ours_full = pl.concat([pl.read_parquet(p) for p in existing])
    ours_full = ours_full.filter(
        (pl.col("date") >= start) & (pl.col("date") <= end)
    )

    if top is not None:
        # Rank by mean daily turnover across the requested window. Filters
        # out illiquid names that would dominate Yahoo failures without
        # actually reflecting "important" stocks.
        ranked = (
            ours_full.group_by("symbol")
            .agg(mean_turnover=pl.col("turnover").mean())
            .sort("mean_turnover", descending=True)
            .head(top)
        )
        sym_list = ranked["symbol"].to_list()
        console.print(f"[dim]top {len(sym_list)} symbols by mean turnover selected[/dim]")
    else:
        sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
        if not sym_list:
            raise typer.BadParameter("--symbols must contain at least one symbol")

    table = Table(title=f"reconcile vs Yahoo  ({ex} {start} → {end}, ±{tolerance_pct}%)",
                  border_style="cyan")
    table.add_column("Symbol", style="bold")
    table.add_column("Rows", justify="right")
    table.add_column("Within tol", justify="right", style="green")
    table.add_column("Max diff %", justify="right")
    table.add_column("Mean diff %", justify="right")

    stats_list = []
    for i, sym in enumerate(sym_list):
        ours = ours_full.filter(pl.col("symbol") == sym).select(["date", "adj_close"])
        if ours.height == 0:
            console.print(f"[yellow]skip[/yellow] {sym}: no rows in adjusted parquet")
            continue
        try:
            ref = fetch_yahoo_adjusted(sym, ex, start, end)
        except YahooFetchError as e:
            console.print(f"[red]yahoo fail[/red] {sym}: {e}")
            continue
        s = reconcile_symbol(sym, ours, ref, tolerance_pct=tolerance_pct)
        stats_list.append(s)
        within_color = "green" if s.pct_within_tol >= 99.0 else "yellow" if s.pct_within_tol >= 95.0 else "red"
        table.add_row(
            sym,
            str(s.rows_compared),
            f"[{within_color}]{s.pct_within_tol:.2f}%[/{within_color}]",
            f"{s.max_abs_diff_pct:.3f}",
            f"{s.mean_abs_diff_pct:.3f}",
        )
        if i < len(sym_list) - 1:
            time.sleep(sleep_between)

    console.print(table)

    if stats_list:
        result = summarize(stats_list)
        body = (
            f"[bold]symbols[/bold]   {len(stats_list)}\n"
            f"[bold]rows[/bold]      {result.overall_rows}\n"
            f"[bold]overall[/bold]   {result.overall_pct_within_tol:.2f}% within ±{tolerance_pct}%"
        )
        color = "green" if result.overall_pct_within_tol >= 99.0 else "yellow"
        console.print(Panel.fit(body, border_style=color))


@app.command("publish-r2")
def publish_r2(
    data_dir: Annotated[
        Path, typer.Option("--data-dir", help="Local parquet root to push")
    ] = DEFAULT_OUT_DIR,
    bucket: Annotated[
        str, typer.Option("--bucket", help="R2 bucket name")
    ] = DEFAULT_R2_BUCKET,
    prefix: Annotated[
        str,
        typer.Option("--prefix", help="Optional key prefix inside the bucket"),
    ] = "",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="List files, do not upload"),
    ] = False,
) -> None:
    """Push partitioned parquet under DATA_DIR to a Cloudflare R2 bucket.

    Credentials come from env: R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY.
    Idempotent: files whose remote ETag matches local md5 are skipped.
    """
    _banner()
    try:
        result = publish_to_r2(
            data_dir,
            bucket=bucket,
            prefix=prefix,
            dry_run=dry_run,
        )
    except PublishR2Error as e:
        console.print(f"[red]publish-r2 failed[/red]: {e}")
        raise typer.Exit(code=1) from e

    total_mb = result.total_bytes / 1024 / 1024
    up_mb = result.uploaded_bytes / 1024 / 1024
    body = (
        f"[bold]bucket[/bold]     {result.bucket}\n"
        f"[bold]files[/bold]      {result.file_count}  ({total_mb:.2f} MB)\n"
        f"[bold]uploaded[/bold]   {result.uploaded_count}  ({up_mb:.2f} MB)\n"
        f"[bold]skipped[/bold]    {result.skipped_count}  [dim](etag match)[/dim]"
    )
    if dry_run:
        body += "\n[yellow]dry-run, nothing uploaded[/yellow]"
    console.print(Panel.fit(body, border_style="green" if not dry_run else "yellow"))


@app.command("export-json")
def export_json(
    prices_dir: Annotated[
        Path, typer.Option("--prices-dir", help="Root of bhavcopy parquet (nse/, bse/)")
    ] = DEFAULT_OUT_DIR,
    actions_dir: Annotated[
        Optional[Path], typer.Option("--actions-dir", help="Directory of actions parquet; omit to skip")
    ] = None,
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Output root; files land under <out>/api/v1/...")
    ] = Path("data/api"),
    exchange: Annotated[
        ExchangeChoice,
        typer.Option("--exchange", "-e", help="Exchange to export", case_sensitive=False),
    ] = ExchangeChoice.BOTH,
    snapshot_years: Annotated[
        str,
        typer.Option(
            "--snapshot-years",
            help="`current` (default), `all`, or comma-separated years. "
            "Past snapshots never change, so the daily cron only rewrites this year.",
        ),
    ] = "current",
    only: Annotated[
        str,
        typer.Option("--only", help="Comma list of ohlcv,snapshot,actions (default all)"),
    ] = "ohlcv,snapshot,actions",
) -> None:
    """Pre-render tej-api free-tier JSON (ohlcv, snapshot, actions) for edge serving."""
    from pipeline.export_json import ExportError, export_api_json

    _banner()
    if snapshot_years == "all":
        years = None
    elif snapshot_years == "current":
        years = [date.today().year]
    else:
        years = [int(y) for y in snapshot_years.split(",") if y.strip()]
    try:
        res = export_api_json(
            prices_dir=prices_dir,
            actions_dir=actions_dir,
            out_dir=out_dir,
            exchanges=_exchanges(exchange),
            snapshot_years=years,
            only=[k.strip() for k in only.split(",") if k.strip()],
        )
    except ExportError as e:
        console.print(f"[red]export-json failed[/red]: {e}")
        raise typer.Exit(code=1) from e
    console.print(Panel.fit(
        f"[bold]out[/bold]        {out_dir}\n"
        f"[bold]ohlcv[/bold]      {res.ohlcv_files} files\n"
        f"[bold]snapshot[/bold]   {res.snapshot_files} files\n"
        f"[bold]actions[/bold]    {res.action_files} files",
        border_style="green",
    ))


universe_app = typer.Typer(
    name="universe",
    help="Point-in-time liquidity universes (survivorship-bias-free).",
    no_args_is_help=True,
)
app.add_typer(universe_app)


@universe_app.command("build")
def universe_build(
    exchange: Annotated[
        ExchangeChoice,
        typer.Option("--exchange", "-e", help="Exchange to build", case_sensitive=False),
    ] = ExchangeChoice.BOTH,
    prices_dir: Annotated[
        Path,
        typer.Option("--prices-dir", help="Root of partitioned bhavcopy parquet"),
    ] = DEFAULT_OUT_DIR,
    out_dir: Annotated[
        Path,
        typer.Option("--out-dir", help="Directory for universe parquet"),
    ] = DEFAULT_UNIVERSE_DIR,
    top_n: Annotated[
        int, typer.Option("--top-n", help="Members kept per rebalance"),
    ] = 500,
) -> None:
    """Monthly top-N by trailing 63-day turnover, one file per exchange.

    Output `<out-dir>/<ex>_liquid.parquet`. Rebuilt from the full price
    history every run; membership on a rebalance date only uses data up to
    that date, so delisted names remain in the months they qualified for.
    """
    from pipeline.export_json import read_bhavcopy
    from pipeline.universe import build_universe

    _banner()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ex in _exchanges(exchange):
        paths = sorted((prices_dir / ex.lower()).rglob("*.parquet"))
        if not paths:
            console.print(f"[red]no bhavcopy parquet under {prices_dir / ex.lower()}[/red]")
            raise typer.Exit(code=1)
        uni = build_universe(read_bhavcopy(paths), ex, top_n=top_n)
        out = out_dir / f"{ex.lower()}_liquid.parquet"
        uni.write_parquet(out, compression="zstd")
        n_rb = uni["rebalance_date"].n_unique() if not uni.is_empty() else 0
        console.print(Panel.fit(
            f"[bold]exchange[/bold]    {ex}\n"
            f"[bold]rebalances[/bold]  {n_rb}\n"
            f"[bold]rows[/bold]        {uni.height}\n"
            f"[bold]out[/bold]         {out}",
            border_style="green",
        ))


@app.command("pull-r2")
def pull_r2(
    prefix: Annotated[
        list[str],
        typer.Option(
            "--prefix",
            help="R2 key prefix to mirror down (repeatable, e.g. `--prefix nse/`)",
        ),
    ],
    data_dir: Annotated[
        Path, typer.Option("--data-dir", help="Local parquet root to pull into")
    ] = DEFAULT_OUT_DIR,
    bucket: Annotated[
        str, typer.Option("--bucket", help="R2 bucket name")
    ] = DEFAULT_R2_BUCKET,
) -> None:
    """Mirror R2 parquet under the given prefixes into DATA_DIR.

    Local-wins: files already present on disk are never overwritten. The
    daily cron uses this to seed full price + actions history before the
    all-years derived builds (symbol-history, adjust, metrics), which
    otherwise see only the single day fetched in that run.

    Credentials come from env: R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY.
    """
    _banner()
    try:
        result = pull_from_r2(data_dir, prefixes=list(prefix), bucket=bucket)
    except PublishR2Error as e:
        console.print(f"[red]pull-r2 failed[/red]: {e}")
        raise typer.Exit(code=1) from e

    dl_mb = result.downloaded_bytes / 1024 / 1024
    console.print(Panel.fit(
        f"[bold]bucket[/bold]       {result.bucket}\n"
        f"[bold]listed[/bold]       {result.listed_count}\n"
        f"[bold]downloaded[/bold]   {result.downloaded_count}  ({dl_mb:.2f} MB)\n"
        f"[bold]skipped[/bold]      {result.skipped_count}  [dim](local wins)[/dim]",
        border_style="green",
    ))


@app.command()
def compact(
    year: Annotated[
        int, typer.Option("--year", help="Past year to compact (do not compact the current year)")
    ],
    exchange: Annotated[
        ExchangeChoice,
        typer.Option("--exchange", "-e", help="Exchange to compact", case_sensitive=False),
    ] = ExchangeChoice.BOTH,
    data_dir: Annotated[
        Path, typer.Option("--data-dir", help="Local parquet root")
    ] = DEFAULT_OUT_DIR,
    delete_dailies: Annotated[
        bool,
        typer.Option(
            "--delete-dailies/--keep-dailies",
            help="Remove month=*/date=*.parquet files after writing the rollup",
        ),
    ] = True,
    from_r2: Annotated[
        bool,
        typer.Option(
            "--from-r2",
            help="Download daily files from R2 instead of reading local. Needed on "
            "ephemeral runners that do not carry historical data on disk.",
        ),
    ] = False,
    bucket: Annotated[
        str,
        typer.Option("--bucket", help="R2 bucket name (only used with --from-r2)"),
    ] = DEFAULT_R2_BUCKET,
    refresh: Annotated[
        bool,
        typer.Option(
            "--refresh",
            help="Rebuild the rollup even if it already exists. Use for the "
            "current year so each new daily file lands in the rollup the same "
            "evening it is fetched.",
        ),
    ] = False,
) -> None:
    """Compact one year of daily bhavcopy files into a single rollup parquet.

    Reads `<data-dir>/<ex>/year=YYYY/month=*/date=*.parquet`, writes
    `<data-dir>/<ex>/year=YYYY/<ex>_<YYYY>.parquet`, then optionally deletes
    the daily files. Run this once at the start of each new year for the
    just-closed year, to cut R2 LIST overhead for long-range queries.
    Idempotent: skips if the rollup already exists.
    """
    _banner()
    table = Table(title=f"compact year={year}", border_style="cyan")
    table.add_column("Exchange", style="bold")
    table.add_column("Source", justify="right")
    table.add_column("Rows", justify="right", style="green")
    table.add_column("Rollup", style="dim")
    table.add_column("Action")

    for ex in _exchanges(exchange):
        year_dir = data_dir / ex.lower() / f"year={year}"
        rollup_path = year_dir / f"{ex.lower()}_{year}.parquet"
        if rollup_path.exists() and not refresh:
            table.add_row(
                ex, "-", "-", str(rollup_path), "[yellow]rollup exists, skip[/yellow]"
            )
            continue
        # When refreshing, drop the stale rollup before reading inputs so
        # --from-r2 does not double-count.
        if rollup_path.exists() and refresh:
            rollup_path.unlink()

        if from_r2:
            paths_to_read, n_sources, cleanup = _fetch_year_dailies_from_r2(
                bucket=bucket, exchange=ex, year=year
            )
        else:
            if not year_dir.exists():
                table.add_row(ex, "-", "-", "-", "[yellow]no data[/yellow]")
                continue
            daily_files = sorted(year_dir.glob("month=*/date=*.parquet"))
            paths_to_read = [str(p) for p in daily_files]
            n_sources = len(daily_files)
            cleanup = lambda: None  # noqa: E731

        try:
            if not paths_to_read:
                table.add_row(ex, "0", "0", "-", "[yellow]no dailies[/yellow]")
                continue
            # Rollup + daily files mix schemas (see _read_bhavcopy_mixed);
            # the current-year refresh reads both, so normalise per-file.
            df = _read_bhavcopy_mixed(paths_to_read)
            # Re-add year + month as REAL columns so queries that filter on
            # them work across both the rollup (no hive month=/date= dir) and
            # daily files (where year/month live in the path).
            df = df.with_columns(
                pl.col("date").dt.year().alias("year"),
                pl.col("date").dt.month().alias("month"),
            )
            # When --from-r2 includes the existing rollup AND any new daily
            # files (current-year refresh path), the same (symbol, date) can
            # show up twice. Drop duplicates so we keep one row per cell.
            df = df.unique(subset=["symbol", "date"], keep="first")
            # Sort by (symbol, date) so parquet row-group min/max stats on
            # `symbol` cluster tightly. Single-symbol OHLCV queries can then
            # skip every row group whose symbol range excludes the target,
            # turning a ~30 MB-per-year scan into ~50 kB per year.
            df = df.sort(["symbol", "date"])
            rollup_path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(rollup_path, compression="zstd")
        finally:
            cleanup()

        action = "[green]wrote rollup[/green]"
        if delete_dailies and not from_r2:
            for p in year_dir.glob("month=*/date=*.parquet"):
                p.unlink()
            for month_dir in year_dir.glob("month=*"):
                try:
                    month_dir.rmdir()
                except OSError:
                    pass
            action += " [dim]+ deleted local dailies[/dim]"
        table.add_row(
            ex, f"{n_sources}" if not from_r2 else f"{n_sources} (r2)",
            f"{df.height:,}", str(rollup_path), action,
        )

    console.print(table)


def _fetch_year_dailies_from_r2(
    bucket: str, exchange: str, year: int,
) -> tuple[list[str], int, Any]:
    """Download every bhavcopy key for one year + exchange to a temp dir.

    Includes BOTH the rollup (`year=YYYY/<ex>_<YYYY>.parquet`, if present)
    AND any daily files (`year=YYYY/month=*/date=*.parquet`). The compact
    caller dedupes by (symbol, date) so the rollup's pre-existing rows are
    not double-counted when a daily file overlaps. This shape lets the daily
    cron refresh the current year's rollup safely: today's new daily file
    plus the existing rollup are merged into the next rollup.

    Returns (list-of-local-paths, count, cleanup-callable). The caller must
    invoke cleanup() in a finally block.
    """
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    from pipeline.publish_r2 import _build_client

    s3 = _build_client(None, None, None)
    year_prefix = f"{exchange.lower()}/year={year}/"
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=year_prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                keys.append(obj["Key"])

    tmp = tempfile.mkdtemp(prefix=f"compact-{exchange.lower()}-{year}-")
    tmp_path = Path(tmp)
    local_paths: list[str] = []

    def _download_one(key: str) -> str:
        # Flatten subdirs into the temp dir; we read them all into one
        # frame and dedupe, so original layout does not matter.
        lp = tmp_path / Path(key).name
        s3.download_file(bucket, key, str(lp))
        return str(lp)

    if keys:
        with ThreadPoolExecutor(max_workers=8) as pool:
            local_paths = list(pool.map(_download_one, keys))

    def cleanup() -> None:
        import shutil

        shutil.rmtree(tmp_path, ignore_errors=True)

    return local_paths, len(local_paths), cleanup


@app.command("r2-prune")
def r2_prune(
    prefix: Annotated[
        str,
        typer.Option(
            "--prefix",
            help="R2 key prefix to delete recursively (eg `nse/year=2010/month=`)",
        ),
    ],
    bucket: Annotated[
        str, typer.Option("--bucket", help="R2 bucket name")
    ] = DEFAULT_R2_BUCKET,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="List would-be-deleted keys without deleting")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", help="Skip the are-you-sure prompt for non-dry-run")
    ] = False,
) -> None:
    """Delete every R2 object under a prefix. Use AFTER compacting a year to
    drop the now-redundant daily bhavcopy keys.

    Refuses prefixes without a `/` so a typo cannot wipe the bucket.
    """
    _banner()
    if not dry_run and not yes:
        confirm = typer.confirm(
            f"This will permanently delete every object under r2://{bucket}/{prefix}. Continue?"
        )
        if not confirm:
            raise typer.Exit(code=1)
    try:
        res = prune_r2_prefix(prefix, bucket=bucket, dry_run=dry_run)
    except PublishR2Error as e:
        console.print(f"[red]r2-prune failed[/red] {e}")
        raise typer.Exit(code=1) from e
    console.print(
        Panel.fit(
            f"[bold]bucket[/bold]   {res.bucket}\n"
            f"[bold]prefix[/bold]   {res.prefix}\n"
            f"[bold]deleted[/bold]  {res.deleted_count}"
            + ("\n[yellow]dry-run, nothing deleted[/yellow]" if dry_run else ""),
            border_style="green" if not dry_run else "yellow",
        )
    )


@app.command()
def version() -> None:
    """Print version and exit."""
    console.print(f"tej-bazaar [bold cyan]{__version__}[/bold cyan]")


def _date_from_path(p: Path) -> date:
    # date=YYYY-MM-DD.parquet → YYYY-MM-DD
    stem = p.stem  # "date=2025-04-30"
    return _parse_date(stem.split("=", 1)[1])


def _year_from_actions_filename(p: Path) -> int | None:
    # Annual cron files are named `<exchange>_<YYYY>.parquet`. Range files
    # like `<exchange>_<YYYYMMDD>_<YYYYMMDD>.parquet` are skipped here.
    stem = p.stem
    parts = stem.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit() or len(parts[1]) != 4:
        return None
    return int(parts[1])


def main() -> None:
    app()


if __name__ == "__main__":
    main()
