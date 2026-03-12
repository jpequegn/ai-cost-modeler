"""Comparator: side-by-side architecture comparison with Cherny check."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from costmodel.estimator import CostEstimate, estimate
from costmodel.ledger import ArchStats, CostLedger
from costmodel.models import Architecture


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
    """Result of the Boris Cherny hypothesis check."""

    model_a: str  # typically "opus"
    model_b: str  # typically "haiku"
    cost_a: float  # actual or estimated per-run cost
    cost_b: float  # actual or estimated per-run cost
    cost_ratio: float  # cost_a / cost_b
    retry_rate_a: Optional[float]  # actual retry rate (None = no data)
    retry_rate_b: Optional[float]
    # breakeven: what retry-rate ratio would make them equal cost?
    # if A is 6× more expensive, A wins when retry_b / retry_a > 6
    breakeven_ratio: float
    verdict: str  # "MODEL_A_WINS" | "MODEL_B_WINS" | "INSUFFICIENT_DATA" | "DEPENDS"
    verdict_detail: str
    data_source: str  # "estimated" | "actual" | "mixed"
    runs_a: int
    runs_b: int


@dataclass
class ComparisonReport:
    rows: list[ArchRow]
    baseline_name: Optional[str]
    cherny: Optional[ChernyCheckResult]


class ArchitectureComparator:
    """Compare multiple architectures on estimated and actual cost."""

    MINIMUM_RUNS_FOR_CHERNY = 10

    def __init__(self, ledger: Optional[CostLedger] = None) -> None:
        self._architectures: list[Architecture] = []
        self._baseline_name: Optional[str] = None
        self._ledger = ledger

    def add(self, arch: Architecture) -> "ArchitectureComparator":
        self._architectures.append(arch)
        return self

    def set_baseline(self, name: str) -> "ArchitectureComparator":
        """Mark an architecture as the cost baseline for ratio comparison."""
        self._baseline_name = name
        return self

    def compare(self) -> ComparisonReport:
        """Run the comparison and return a full report."""
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

        # Compute baseline ratios
        baseline_row = None
        if self._baseline_name:
            for r in rows:
                if r.name == self._baseline_name:
                    baseline_row = r
                    break

        if baseline_row:
            baseline_cost = (
                baseline_row.act_per_run
                if baseline_row.act_per_run is not None
                else baseline_row.est_per_run
            )
            for r in rows:
                if r.name == self._baseline_name:
                    r.vs_baseline_ratio = None  # IS the baseline
                elif baseline_cost > 0:
                    row_cost = r.act_per_run if r.act_per_run is not None else r.est_per_run
                    r.vs_baseline_ratio = row_cost / baseline_cost

        # Sort by estimated cost (cheapest first, baseline last)
        def sort_key(r: ArchRow) -> float:
            if r.name == self._baseline_name:
                return float("inf")
            return r.est_per_run

        rows.sort(key=sort_key)

        cherny = self._cherny_check(rows)
        return ComparisonReport(rows=rows, baseline_name=self._baseline_name, cherny=cherny)

    def _cherny_check(self, rows: list[ArchRow]) -> Optional[ChernyCheckResult]:
        """Identify the most and least expensive non-baseline models and run the check."""
        if len(rows) < 2:
            return None

        non_baseline = [r for r in rows if r.name != self._baseline_name]
        if len(non_baseline) < 2:
            return None

        # Compare most expensive vs cheapest (by estimate)
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
        breakeven_ratio = cost_ratio  # model_b retry rate / model_a rate must exceed this

        # Determine verdict
        retry_rate_a: Optional[float] = None
        retry_rate_b: Optional[float] = None

        if not enough_data_a or not enough_data_b:
            verdict = "INSUFFICIENT_DATA"
            verdict_detail = (
                f"Need ≥{self.MINIMUM_RUNS_FOR_CHERNY} runs per model for statistical validity. "
                f"Have: {runs_a} for {most_expensive.name}, {runs_b} for {cheapest.name}."
            )
        else:
            # We have enough real data — just compare actual costs
            if cost_a <= cost_b:
                verdict = "MODEL_A_WINS"
                verdict_detail = (
                    f"{most_expensive.name} (${cost_a:.4f}/run) is actually cheaper than "
                    f"{cheapest.name} (${cost_b:.4f}/run) — Cherny hypothesis CONFIRMED ✓"
                )
            else:
                if cost_ratio < 2.0:
                    verdict = "DEPENDS"
                    verdict_detail = (
                        f"Only {cost_ratio:.1f}× price difference. "
                        f"Better model may win with modest retry reduction."
                    )
                else:
                    verdict = "MODEL_B_WINS"
                    verdict_detail = (
                        f"{cheapest.name} (${cost_b:.4f}/run) is cheaper. "
                        f"{most_expensive.name} would need {cost_ratio:.1f}× fewer retries to win."
                    )

        return ChernyCheckResult(
            model_a=most_expensive.name,
            model_b=cheapest.name,
            cost_a=cost_a,
            cost_b=cost_b,
            cost_ratio=cost_ratio,
            retry_rate_a=retry_rate_a,
            retry_rate_b=retry_rate_b,
            breakeven_ratio=breakeven_ratio,
            verdict=verdict,
            verdict_detail=verdict_detail,
            data_source=data_source,
            runs_a=runs_a,
            runs_b=runs_b,
        )
