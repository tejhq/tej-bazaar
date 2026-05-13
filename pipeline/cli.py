"""tej-bazaar CLI: fetch, transform, write NSE/BSE bhavcopy as parquet.

Commands:
    tej-bazaar fetch DATE [--exchange NSE|BSE|both]
    tej-bazaar backfill --from D --to D [--exchange NSE|BSE|both]
    tej-bazaar info [--data-dir PATH]
    tej-bazaar actions fetch --from D --to D [--exchange NSE|BSE|both]
    tej-bazaar version
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Annotated

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
from pipeline.push import partition_path, write_partitioned
from pipeline.transform import transform

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUT_DIR = Path("data/out")
DEFAULT_ACTIONS_CACHE_DIR = Path("data/raw/actions")
DEFAULT_ACTIONS_OUT_DIR = Path("data/out/actions")
DEFAULT_PRICES_ADJUSTED_DIR = Path("data/out/prices_adjusted")


class ExchangeChoice(str, Enum):
    NSE = "NSE"
    BSE = "BSE"
    BOTH = "both"


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
    help="Free, open EOD market data for India — NSE & BSE.",
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
                console.print(f"[yellow]skip[/yellow] {ex} {d} — {e}")
                progress.advance(task)
                continue
            except BhavcopyFetchError as e:
                console.print(f"[red]error[/red] {ex} {d} — {e}")
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
                except Exception as e:  # noqa: BLE001 — keep loop alive on any per-day error
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
) -> None:
    """Push partitioned parquet under DATA_DIR to a HuggingFace dataset repo."""
    _banner()
    try:
        result = publish_to_hf(
            data_dir,
            repo_id=repo,
            commit_message=message,
            dry_run=dry_run,
        )
    except PublishError as e:
        console.print(f"[red]publish failed[/red] — {e}")
        raise typer.Exit(code=1) from e

    mb = result.total_bytes / 1024 / 1024
    body = (
        f"[bold]repo[/bold]   {result.repo_id}\n"
        f"[bold]files[/bold]  {result.file_count}\n"
        f"[bold]size[/bold]   {mb:.2f} MB"
    )
    if dry_run:
        body += "\n[yellow]dry-run — nothing uploaded[/yellow]"
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
    year: Annotated[int, typer.Option("--year", help="Calendar year to adjust")],
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
    """
    _banner()
    exchanges = _exchanges(exchange)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = Table(title="actions adjust", border_style="cyan")
    summary.add_column("Exchange", style="bold")
    summary.add_column("Price rows", justify="right")
    summary.add_column("Actions", justify="right")
    summary.add_column("Adjusted rows", justify="right", style="green")
    summary.add_column("Output", style="dim")

    for ex in exchanges:
        prices_glob = list((prices_dir / ex.lower()).rglob(f"year={year}/**/*.parquet"))
        if not prices_glob:
            console.print(f"[yellow]skip[/yellow] {ex}: no bhavcopy parquet under "
                          f"{prices_dir / ex.lower()}/year={year}")
            continue

        prices = pl.concat([pl.read_parquet(p) for p in prices_glob])

        actions_path = actions_dir / f"{ex.lower()}_{year}.parquet"
        if not actions_path.exists():
            console.print(f"[yellow]skip[/yellow] {ex}: actions parquet missing at {actions_path}")
            continue
        actions = pl.read_parquet(actions_path)

        factors = compute_action_factors(actions, prices)
        adjusted = back_adjust(prices, factors)
        out_path = out_dir / f"{ex.lower()}_{year}.parquet"
        adjusted.write_parquet(out_path)
        summary.add_row(ex, str(prices.height), str(actions.height),
                        str(adjusted.height), str(out_path))

    console.print(summary)


@app.command()
def version() -> None:
    """Print version and exit."""
    console.print(f"tej-bazaar [bold cyan]{__version__}[/bold cyan]")


def _date_from_path(p: Path) -> date:
    # date=YYYY-MM-DD.parquet → YYYY-MM-DD
    stem = p.stem  # "date=2025-04-30"
    return _parse_date(stem.split("=", 1)[1])


def main() -> None:
    app()


if __name__ == "__main__":
    main()
