"""Comparator: side-by-side architecture comparison with Cherny check."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from costmodel.estimator import CostEstimate, estimate
from costmodel.ledger import ArchStats, CostLedger
from costmodel.models import Architecture


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ArchRow:
    """One row in the comparison table."""

    name: str
    est_per_run: float
    act_per_run: Optional[float]  # None if no ledger data
    per_1000_day: float  # estimated cost per 1000 runs/day
    run_count: int
    vs_baseline_ratio: Optional[float]  # cost ratio vs baseline (None if IS baseline)
    estimate: CostEstimate
    stats: Optional[ArchStats]


@dataclass
class ChernyCheckResult:
    """Result of the Boris Cherny hypothesis check (opus vs haiku)."""

    model_a: str  # typically "opus" / most expensive
    model_b: str  # typically "haiku" / cheapest
    cost_a: float  # actual or estimated per-run cost
    cost_b: float  # actual or estimated per-run cost
    cost_ratio: float  # cost_a / cost_b
    retry_rate_a: Optional[float]  # actual retry rate from ledger (None = no data)
    retry_rate_b: Optional[float]
    effective_cost_a: Optional[float]  # cost_a * (1 + retry_rate_a)
    effective_cost_b: Optional[float]  # cost_b * (1 + retry_rate_b)
    # breakeven: what retry-rate ratio would make them equal cost?
    # if A is N× more expensive, A wins when retry_b / retry_a > N
    breakeven_ratio: float
    verdict: str  # "MODEL_A_WINS" | "MODEL_B_WINS" | "INSUFFICIENT_DATA" | "DEPENDS"
    verdict_detail: str
    data_source: str  # "estimated" | "actual" | "mixed"
    runs_a: int
    runs_b: int


@dataclass
class ComparisonReport:
    """Full comparison report: table rows + Cherny result."""

    rows: list[ArchRow]
    baseline_name: Optional[str]
    cherny: Optional[ChernyCheckResult]
    task_label: str = ""

    # ------------------------------------------------------------------
    # Public render API
    # ------------------------------------------------------------------

    def render(self, console: Optional[Console] = None) -> None:
        """Render the comparison table and Cherny check to the console."""
        if console is None:
            console = Console()

        _render_comparison_table(self, console)
        if self.cherny:
            _render_cherny_check(self.cherny, console)


# ---------------------------------------------------------------------------
# Helpers: formatting
# ---------------------------------------------------------------------------


def _fmt_usd(amount: float) -> str:
    """Format a USD amount compactly."""
    if amount == 0:
        return "$0.00"
    if amount < 0.001:
        return f"${amount:.6f}"
    if amount < 0.01:
        return f"${amount:.4f}"
    if amount < 1.0:
        return f"${amount:.4f}"
    if amount < 1_000:
        return f"${amount:,.2f}"
    return f"${amount:,.0f}"


def _fmt_ratio(ratio: float) -> str:
    """Format a cost ratio. Values ≥1 become e.g. '3,333×'."""
    if ratio >= 10_000:
        return f"{ratio:,.0f}×"
    if ratio >= 1_000:
        return f"{ratio:,.0f}×"
    if ratio >= 100:
        return f"{ratio:,.0f}×"
    if ratio >= 10:
        return f"{ratio:.0f}×"
    return f"{ratio:.1f}×"


def _fmt_pct(rate: Optional[float]) -> str:
    if rate is None:
        return "N/A"
    return f"{rate * 100:.1f}%"


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _render_comparison_table(report: ComparisonReport, console: Console) -> None:
    """Render the side-by-side architecture comparison table."""
    title = "Architecture Comparison"
    if report.task_label:
        title += f" — {report.task_label}"

    console.print()
    console.print(f"[bold]{title}[/bold]")
    console.print()

    tbl = Table(box=box.SIMPLE_HEAVY, show_footer=False)
    tbl.add_column("Architecture", style="cyan", min_width=26)
    tbl.add_column("Est/run", justify="right", style="yellow")
    tbl.add_column("Act/run", justify="right")
    tbl.add_column("At 1k/day", justify="right")

    show_vs = report.baseline_name is not None
    if show_vs:
        # Column header: "vs <baseline_name>" or just "vs ↑"
        vs_header = "vs ↑"
        tbl.add_column(vs_header, justify="right", style="magenta")

    for row in report.rows:
        act_str: Text | str
        if row.act_per_run is not None:
            act_str = _fmt_usd(row.act_per_run)
        else:
            act_str = Text("N/A", style="dim")

        if show_vs:
            if row.name == report.baseline_name:
                vs_str: Text | str = Text("baseline", style="dim")
            elif row.vs_baseline_ratio is not None:
                # Show how many times cheaper this arch is vs the baseline
                vs_str = _fmt_ratio(row.vs_baseline_ratio)
            else:
                vs_str = Text("—", style="dim")

            tbl.add_row(
                row.name,
                _fmt_usd(row.est_per_run),
                act_str,
                _fmt_usd(row.per_1000_day),
                vs_str,
            )
        else:
            tbl.add_row(
                row.name,
                _fmt_usd(row.est_per_run),
                act_str,
                _fmt_usd(row.per_1000_day),
            )

    console.print(tbl)


def _render_cherny_check(ch: ChernyCheckResult, console: Console) -> None:
    """Render the Cherny hypothesis check block."""
    # Derive short labels (e.g. "opus", "haiku") from full arch names
    def _short(name: str) -> str:
        for kw in ("opus", "haiku", "sonnet"):
            if kw in name.lower():
                return kw
        return name

    label_a = _short(ch.model_a)
    label_b = _short(ch.model_b)

    console.print()
    console.print(
        f"[bold]Cherny Hypothesis Check[/bold] "
        f"({label_a} vs {label_b}):"
    )

    console.print(
        f"  Opus cost/call:    [yellow]{_fmt_usd(ch.cost_a)}[/yellow]  "
        f"Haiku cost/call: [yellow]{_fmt_usd(ch.cost_b)}[/yellow]"
    )

    # Retry rates
    if ch.retry_rate_a is not None and ch.retry_rate_b is not None:
        console.print(
            f"  Opus retry rate:   [yellow]{_fmt_pct(ch.retry_rate_a)}[/yellow]    "
            f"Haiku retry rate: [yellow]{_fmt_pct(ch.retry_rate_b)}[/yellow]"
        )
    else:
        console.print(
            f"  Retry rates:       [dim]not available (no ledger data)[/dim]"
        )

    # Effective costs
    if ch.effective_cost_a is not None and ch.effective_cost_b is not None:
        console.print(
            f"  Opus effective:    [yellow]{_fmt_usd(ch.effective_cost_a)}[/yellow]  "
            f"Haiku effective:  [yellow]{_fmt_usd(ch.effective_cost_b)}[/yellow]"
        )

    # Verdict
    if ch.verdict == "INSUFFICIENT_DATA":
        console.print(f"  Verdict:           [yellow]⚠ {ch.verdict_detail}[/yellow]")
    elif ch.verdict == "MODEL_A_WINS":
        console.print(f"  Verdict:           [green]✓ {ch.verdict_detail}[/green]")
    elif ch.verdict == "MODEL_B_WINS":
        console.print(f"  Verdict:           [red]⚠ {ch.verdict_detail}[/red]")
    else:
        console.print(f"  Verdict:           [yellow]~ {ch.verdict_detail}[/yellow]")

    console.print()


# ---------------------------------------------------------------------------
# Main comparator class
# ---------------------------------------------------------------------------


class ArchitectureComparator:
    """Compare multiple architectures on estimated and actual cost."""

    MINIMUM_RUNS_FOR_CHERNY = 10

    def __init__(self, ledger: Optional[CostLedger] = None) -> None:
        self._architectures: list[Architecture] = []
        self._baseline_name: Optional[str] = None
        self._ledger = ledger

    def add(self, arch: Architecture) -> "ArchitectureComparator":
        """Add an architecture to the comparison set."""
        self._architectures.append(arch)
        return self

    def set_baseline(self, name: str) -> "ArchitectureComparator":
        """Mark an architecture as the cost baseline for ratio comparison."""
        self._baseline_name = name
        return self

    def compare(
        self,
        baseline_cost: Optional[float] = None,
        task_label: str = "",
    ) -> ComparisonReport:
        """Run the comparison and return a full report.

        Args:
            baseline_cost: Override the baseline cost in USD/run (e.g. 20.00
                for Anthropic's commercial review). When provided, this overrides
                the baseline architecture's estimated/actual cost for ratio
                calculations only.
            task_label: Short string shown in the report title (e.g. "code-review task").

        Returns:
            ComparisonReport with rows sorted cheapest-first (baseline last).
        """
        rows: list[ArchRow] = []

        # Build each row
        for arch in self._architectures:
            est = estimate(arch)
            stats: Optional[ArchStats] = None
            act_per_run: Optional[float] = None

            if self._ledger:
                stats = self._ledger.architecture_stats(arch.name)
                if stats and stats.run_count > 0:
                    act_per_run = stats.avg_cost_usd

            rows.append(
                ArchRow(
                    name=arch.name,
                    est_per_run=est.per_run_usd,
                    act_per_run=act_per_run,
                    per_1000_day=est.per_run_usd * 1000,
                    run_count=stats.run_count if stats else 0,
                    vs_baseline_ratio=None,  # filled in below
                    estimate=est,
                    stats=stats,
                )
            )

        # Determine the effective baseline cost
        effective_baseline_cost: Optional[float] = baseline_cost

        if effective_baseline_cost is None and self._baseline_name:
            for r in rows:
                if r.name == self._baseline_name:
                    effective_baseline_cost = (
                        r.act_per_run
                        if r.act_per_run is not None
                        else r.est_per_run
                    )
                    break

        # Compute baseline ratios: how many times more expensive is the baseline?
        # (baseline_cost / row_cost = "N× cheaper than baseline")
        if effective_baseline_cost and effective_baseline_cost > 0:
            for r in rows:
                if r.name == self._baseline_name:
                    r.vs_baseline_ratio = None  # IS the baseline
                else:
                    row_cost = r.act_per_run if r.act_per_run is not None else r.est_per_run
                    if row_cost > 0:
                        r.vs_baseline_ratio = effective_baseline_cost / row_cost
                    else:
                        r.vs_baseline_ratio = None

        # Sort by estimated cost (cheapest first, baseline last)
        def sort_key(r: ArchRow) -> float:
            if r.name == self._baseline_name:
                return float("inf")
            return r.est_per_run

        rows.sort(key=sort_key)

        cherny = self._cherny_check(rows)
        return ComparisonReport(
            rows=rows,
            baseline_name=self._baseline_name,
            cherny=cherny,
            task_label=task_label,
        )

    # ------------------------------------------------------------------
    # Private: Cherny check
    # ------------------------------------------------------------------

    def _cherny_check(self, rows: list[ArchRow]) -> Optional[ChernyCheckResult]:
        """Identify the opus and haiku architectures (or most/least expensive
        non-baseline) and run the Cherny hypothesis check with retry rates from ledger."""
        if len(rows) < 2:
            return None

        non_baseline = [r for r in rows if r.name != self._baseline_name]
        if len(non_baseline) < 2:
            return None

        # Prefer to compare the single-model opus vs haiku architectures.
        # Look for rows whose names contain "opus" and "haiku".
        opus_rows = [r for r in non_baseline if "opus" in r.name.lower()]
        haiku_rows = [r for r in non_baseline if "haiku" in r.name.lower()]

        if opus_rows and haiku_rows:
            # Among opus rows, pick the single-model one if available; else the cheapest
            if len(opus_rows) == 1:
                most_expensive = opus_rows[0]
            else:
                single_opus = [r for r in opus_rows if "single" in r.name.lower() or "agent" not in r.name.lower()]
                most_expensive = single_opus[0] if single_opus else min(opus_rows, key=lambda r: r.est_per_run)

            # Among haiku rows pick cheapest (should only be one)
            cheapest = min(haiku_rows, key=lambda r: r.est_per_run)
        else:
            # Fall back: compare most expensive vs cheapest (by estimate)
            cheapest = min(non_baseline, key=lambda r: r.est_per_run)
            most_expensive = max(non_baseline, key=lambda r: r.est_per_run)

        if cheapest.name == most_expensive.name:
            return None

        # Use actual data if available, otherwise estimates
        cost_a = (
            most_expensive.act_per_run
            if most_expensive.act_per_run is not None
            else most_expensive.est_per_run
        )
        cost_b = (
            cheapest.act_per_run
            if cheapest.act_per_run is not None
            else cheapest.est_per_run
        )
        runs_a = most_expensive.run_count
        runs_b = cheapest.run_count

        has_actual_a = most_expensive.act_per_run is not None
        has_actual_b = cheapest.act_per_run is not None
        enough_data_a = runs_a >= self.MINIMUM_RUNS_FOR_CHERNY
        enough_data_b = runs_b >= self.MINIMUM_RUNS_FOR_CHERNY

        if has_actual_a and has_actual_b:
            data_source = "actual"
        elif has_actual_a or has_actual_b:
            data_source = "mixed"
        else:
            data_source = "estimated"

        cost_ratio = cost_a / cost_b if cost_b > 0 else float("inf")
        breakeven_ratio = cost_ratio

        # Fetch retry rates from ledger if available
        retry_rate_a: Optional[float] = None
        retry_rate_b: Optional[float] = None
        effective_cost_a: Optional[float] = None
        effective_cost_b: Optional[float] = None

        if self._ledger:
            retry_rate_a = self._ledger.retry_rate_for_architecture(most_expensive.name)
            retry_rate_b = self._ledger.retry_rate_for_architecture(cheapest.name)

        if retry_rate_a is not None and retry_rate_b is not None:
            # Effective cost = nominal cost × (1 + retry_rate)
            effective_cost_a = cost_a * (1.0 + retry_rate_a)
            effective_cost_b = cost_b * (1.0 + retry_rate_b)

        # Determine verdict
        if not enough_data_a or not enough_data_b:
            verdict = "INSUFFICIENT_DATA"
            verdict_detail = (
                f"Need ≥{self.MINIMUM_RUNS_FOR_CHERNY} runs per model. "
                f"Have: {runs_a} for {most_expensive.name}, {runs_b} for {cheapest.name}."
            )
        else:
            # Use effective cost if we have retry data, otherwise nominal
            compare_a = effective_cost_a if effective_cost_a is not None else cost_a
            compare_b = effective_cost_b if effective_cost_b is not None else cost_b

            if compare_a <= compare_b:
                verdict = "MODEL_A_WINS"
                verdict_detail = (
                    f"{most_expensive.name} is actually cheaper once retries are counted — "
                    f"Cherny hypothesis CONFIRMED ✓"
                )
            else:
                eff_ratio = compare_a / compare_b if compare_b > 0 else float("inf")
                if eff_ratio < 2.0:
                    verdict = "DEPENDS"
                    verdict_detail = (
                        f"Only {eff_ratio:.1f}× effective difference. "
                        f"Better model may win with modest retry reduction."
                    )
                else:
                    # Determine a label for model_b (haiku) 
                    model_b_label = cheapest.name.split("-")[-1] if "-" in cheapest.name else cheapest.name
                    verdict = "MODEL_B_WINS"
                    verdict_detail = (
                        f"{model_b_label.capitalize()} still cheaper — "
                        f"{most_expensive.name.split('-')[-1].capitalize() if '-' in most_expensive.name else most_expensive.name} "
                        f"justified only for complex tasks"
                    )

        return ChernyCheckResult(
            model_a=most_expensive.name,
            model_b=cheapest.name,
            cost_a=cost_a,
            cost_b=cost_b,
            cost_ratio=cost_ratio,
            retry_rate_a=retry_rate_a,
            retry_rate_b=retry_rate_b,
            effective_cost_a=effective_cost_a,
            effective_cost_b=effective_cost_b,
            breakeven_ratio=breakeven_ratio,
            verdict=verdict,
            verdict_detail=verdict_detail,
            data_source=data_source,
            runs_a=runs_a,
            runs_b=runs_b,
        )
