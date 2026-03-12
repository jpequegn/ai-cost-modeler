"""Reporter: generate markdown cost reports from ledger data."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from costmodel.ledger import CostLedger, RunSummary


def _format_usd(amount: float) -> str:
    if amount == 0:
        return "$0.00"
    if amount < 0.001:
        return f"${amount:.6f}"
    if amount < 0.01:
        return f"${amount:.4f}"
    return f"${amount:.4f}"


def _format_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def generate_report(
    ledger: CostLedger,
    limit: int = 50,
    architecture_name: Optional[str] = None,
) -> str:
    """Generate a markdown cost report from ledger data.

    Args:
        ledger: The CostLedger instance to pull data from.
        limit: Number of recent runs to include.
        architecture_name: Filter to a specific architecture (optional).

    Returns:
        Markdown-formatted report as a string.
    """
    runs = ledger.recent_runs(limit=limit, architecture_name=architecture_name)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    title_suffix = f" — {architecture_name}" if architecture_name else ""
    lines: list[str] = [
        f"# AI Cost Report{title_suffix}",
        f"Generated: {now}",
        f"Showing: last {min(limit, len(runs))} runs",
        "",
    ]

    if not runs:
        lines += [
            "## No Data",
            "",
            "No runs found in the ledger. Start tracking runs with `cost track`.",
            "",
        ]
        return "\n".join(lines)

    # ── Summary section ──────────────────────────────────────────────────────
    costs = [r.total_cost_usd for r in runs]
    tokens = [r.total_tokens for r in runs]
    total_cost = sum(costs)
    avg_cost = total_cost / len(runs)
    sorted_costs = sorted(costs)
    sorted_tokens = sorted(tokens)
    n = len(runs)

    most_expensive_run = max(runs, key=lambda r: r.total_cost_usd)
    cheapest_run = min(runs, key=lambda r: r.total_cost_usd)

    lines += [
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total spent | {_format_usd(total_cost)} |",
        f"| Avg per run | {_format_usd(avg_cost)} |",
        f"| Most expensive run | {_format_usd(most_expensive_run.total_cost_usd)} (run: `{most_expensive_run.run_id}`) |",
        f"| Cheapest run | {_format_usd(cheapest_run.total_cost_usd)} (run: `{cheapest_run.run_id}`) |",
        f"| Total runs | {n} |",
        "",
    ]

    # ── Token distribution ────────────────────────────────────────────────────
    if tokens:
        p50_tokens = sorted_tokens[int(n * 0.50)]
        p95_tokens = sorted_tokens[min(int(n * 0.95), n - 1)]
        lines += [
            "## Token Distribution",
            "",
            f"| Percentile | Tokens |",
            f"|------------|--------|",
            f"| p50 | {_format_tokens(p50_tokens)} |",
            f"| p95 | {_format_tokens(p95_tokens)} |",
            f"| max | {_format_tokens(max(tokens))} |",
            f"| avg | {_format_tokens(int(sum(tokens) / n))} |",
            "",
        ]

    # ── Cost distribution ─────────────────────────────────────────────────────
    p50_cost = sorted_costs[int(n * 0.50)]
    p95_cost = sorted_costs[min(int(n * 0.95), n - 1)]

    lines += [
        "## Cost Distribution",
        "",
        f"| Percentile | Cost |",
        f"|------------|------|",
        f"| p50 | {_format_usd(p50_cost)} |",
        f"| p95 | {_format_usd(p95_cost)} |",
        f"| max | {_format_usd(max(costs))} |",
        f"| min | {_format_usd(min(costs))} |",
        "",
    ]

    # ── By architecture ────────────────────────────────────────────────────────
    arch_groups: dict[str, list[RunSummary]] = {}
    for r in runs:
        arch = r.architecture_name or "(unknown)"
        arch_groups.setdefault(arch, []).append(r)

    if len(arch_groups) > 1 or (
        len(arch_groups) == 1 and "(unknown)" not in arch_groups
    ):
        lines += [
            "## By Architecture",
            "",
            "| Architecture | Runs | Total Cost | Avg/Run |",
            "|-------------|------|-----------|---------|",
        ]
        for arch, arch_runs in sorted(arch_groups.items(), key=lambda x: -sum(r.total_cost_usd for r in x[1])):
            arch_total = sum(r.total_cost_usd for r in arch_runs)
            arch_avg = arch_total / len(arch_runs)
            lines.append(
                f"| {arch} | {len(arch_runs)} | {_format_usd(arch_total)} | {_format_usd(arch_avg)} |"
            )
        lines.append("")

    # ── Run log ───────────────────────────────────────────────────────────────
    lines += [
        "## Run Log",
        "",
        "| Run ID | Architecture | Cost | Tokens | Completed |",
        "|--------|-------------|------|--------|-----------|",
    ]
    for r in runs:
        arch = r.architecture_name or "(unknown)"
        lines.append(
            f"| `{r.run_id[:12]}` | {arch} | {_format_usd(r.total_cost_usd)} "
            f"| {_format_tokens(r.total_tokens)} | {r.completed_at[:19]} |"
        )
    lines.append("")

    lines += [
        "---",
        "_Generated by [ai-cost-modeler](https://github.com/jpequegn/ai-cost-modeler)_",
        "",
    ]

    return "\n".join(lines)


def save_report(
    report: str,
    output_path: str | Path,
) -> None:
    """Write a report string to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
