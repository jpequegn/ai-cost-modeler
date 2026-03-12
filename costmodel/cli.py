"""CLI entrypoint for the AI Cost Modeler."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_usd(amount: float) -> str:
    """Format a USD amount nicely."""
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


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _load_arch(arch_value: str):
    """Resolve --arch to an Architecture object.

    Accepts:
    - a built-in name (e.g. "single-agent-opus")
    - a path to a YAML file (e.g. "my-arch.yaml")
    """
    from costmodel.models import get_architecture
    from costmodel.estimator import load_architecture_from_yaml

    # Try YAML file first (if it looks like a path or has .yaml/.yml extension)
    p = Path(arch_value)
    if p.suffix in (".yaml", ".yml") or p.exists():
        if not p.exists():
            console.print(f"[red]Error:[/red] File not found: {arch_value}")
            sys.exit(1)
        return load_architecture_from_yaml(p)

    # Try built-in
    arch = get_architecture(arch_value)
    if arch is not None:
        return arch

    console.print(
        f"[red]Error:[/red] Unknown architecture '{arch_value}'.\n"
        "Use a built-in name or a .yaml file path.\n"
        "Built-in architectures: [cyan]cost estimate --list-archs[/cyan]"
    )
    sys.exit(1)


def _get_ledger():
    """Return a CostLedger instance (creates DB if needed)."""
    from costmodel.ledger import CostLedger
    return CostLedger()


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option()
def cli() -> None:
    """AI Cost Modeler — estimate and track AI inference costs.

    Model architecture costs before you build, track actuals while you run,
    and validate the 'use the best model' hypothesis with real data.
    """


# ---------------------------------------------------------------------------
# cost estimate
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--arch", "arch_value", default=None, help=(
    "Architecture name (built-in) or path to a YAML file."
))
@click.option("--model", default=None, help="Model shortname (e.g. opus, sonnet, haiku).")
@click.option("--task-type", default=None, help=(
    "Task type for heuristic token estimates "
    "(code-review, code-generation, summarization, planning, tool-call, chat)."
))
@click.option("--input-tokens", type=int, default=None, help="Exact input token count.")
@click.option("--output-tokens", type=int, default=None, help="Exact output token count.")
@click.option("--cached-tokens", type=int, default=0, help="Cached input tokens.")
@click.option("--runs", type=int, default=None, help="Show cost at this run volume.")
@click.option("--list-archs", is_flag=True, default=False, help="List built-in architectures.")
def estimate(
    arch_value: Optional[str],
    model: Optional[str],
    task_type: Optional[str],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    cached_tokens: int,
    runs: Optional[int],
    list_archs: bool,
) -> None:
    """Estimate cost for a given architecture or model/token combination.

    Examples:

      cost estimate --arch single-agent-opus

      cost estimate --arch single-agent-opus --task-type code-review --runs 1000

      cost estimate --model opus --input-tokens 8000 --output-tokens 800

      cost estimate --arch my-arch.yaml
    """
    from costmodel.estimator import estimate as do_estimate
    from costmodel.estimator import estimate_from_tokens, estimate_from_task_type
    from costmodel.models import list_architectures

    if list_archs:
        console.print("\n[bold]Built-in architectures:[/bold]")
        for name in list_architectures():
            console.print(f"  [cyan]{name}[/cyan]")
        console.print()
        return

    # Resolve the estimate
    est = None

    if arch_value:
        arch = _load_arch(arch_value)
        # Override task-type tokens if requested
        if task_type and not input_tokens:
            from costmodel.pricing import TASK_INPUT_TOKENS, TASK_OUTPUT_TOKENS
            for stage in arch.stages:
                for call in stage.calls:
                    if task_type in TASK_INPUT_TOKENS:
                        call.input_tokens = TASK_INPUT_TOKENS[task_type]
                        call.output_tokens = output_tokens or TASK_OUTPUT_TOKENS[task_type]
        elif input_tokens is not None:
            for stage in arch.stages:
                for call in stage.calls:
                    call.input_tokens = input_tokens
                    if output_tokens is not None:
                        call.output_tokens = output_tokens
                    call.cached_input_tokens = cached_tokens
        est = do_estimate(arch)

    elif model:
        if input_tokens is not None and output_tokens is not None:
            est = estimate_from_tokens(model, input_tokens, output_tokens, cached_tokens)
        elif task_type:
            est = estimate_from_task_type(model, task_type)
        elif input_tokens is not None:
            est = estimate_from_tokens(model, input_tokens, output_tokens or 400, cached_tokens)
        else:
            console.print(
                "[red]Error:[/red] Provide --task-type or --input-tokens (with --model)."
            )
            sys.exit(1)

    else:
        console.print(
            "[red]Error:[/red] Provide --arch or --model.\n"
            "Run [cyan]cost estimate --help[/cyan] for usage."
        )
        sys.exit(1)

    # ── Print results ────────────────────────────────────────────────────────
    console.print()
    console.print(f"[bold]Cost Estimate:[/bold] [cyan]{est.architecture_name}[/cyan]")
    console.print()

    # Per-stage breakdown
    if len(est.per_stage) > 1:
        stage_table = Table(
            title="Stage Breakdown",
            box=box.SIMPLE_HEAVY,
            show_footer=False,
        )
        stage_table.add_column("Stage", style="cyan")
        stage_table.add_column("In Tokens", justify="right")
        stage_table.add_column("Out Tokens", justify="right")
        stage_table.add_column("Cost/Run", justify="right", style="yellow")

        for s in est.per_stage:
            stage_table.add_row(
                s.stage_name,
                _fmt_tokens(s.input_tokens),
                _fmt_tokens(s.output_tokens),
                _fmt_usd(s.cost_usd),
            )
        console.print(stage_table)
        console.print()

    # Summary table
    summary_table = Table(
        title="Cost Summary",
        box=box.ROUNDED,
        show_footer=False,
    )
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", justify="right", style="green")

    summary_table.add_row("Per run", _fmt_usd(est.per_run_usd))
    summary_table.add_row("Per 1,000 runs", _fmt_usd(est.per_1000_runs_usd))
    summary_table.add_row("Per 10,000 runs", _fmt_usd(est.per_run_usd * 10_000))

    if runs:
        summary_table.add_row(f"Per {runs:,} runs", _fmt_usd(est.per_run_usd * runs))

    # Daily cost estimates at common throughput
    summary_table.add_row("─" * 20, "─" * 15)
    summary_table.add_row("100 runs/day", _fmt_usd(est.per_run_usd * 100))
    summary_table.add_row("1,000 runs/day", _fmt_usd(est.per_run_usd * 1_000))
    summary_table.add_row("10,000 runs/day", _fmt_usd(est.per_run_usd * 10_000))

    # Token totals
    summary_table.add_row("─" * 20, "─" * 15)
    summary_table.add_row("Total input tokens", _fmt_tokens(est.total_input_tokens))
    summary_table.add_row("Total output tokens", _fmt_tokens(est.total_output_tokens))
    summary_table.add_row("Confidence", est.confidence)

    console.print(summary_table)

    if est.notes:
        console.print()
        for note in est.notes:
            console.print(f"  [dim]ℹ {note}[/dim]")

    console.print()


# ---------------------------------------------------------------------------
# cost compare
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--archs", default=None, help=(
    "Comma-separated list of architecture names to compare "
    "(e.g. haiku,sonnet,opus or single-agent-haiku,3-agent-sonnet)."
))
@click.option("--baseline", default=None, help=(
    "Name of the baseline architecture for ratio comparison "
    "(e.g. 'anthropic-code-review')."
))
@click.option("--task-type", default=None, help=(
    "Override task type for token heuristics "
    "(code-review, code-generation, etc.)."
))
@click.option("--with-ledger", is_flag=True, default=False, help=(
    "Include actual run data from the local ledger."
))
def compare(
    archs: Optional[str],
    baseline: Optional[str],
    task_type: Optional[str],
    with_ledger: bool,
) -> None:
    """Compare multiple architectures side-by-side on cost.

    Examples:

      cost compare --archs haiku,sonnet,opus

      cost compare --archs haiku,sonnet,opus --baseline anthropic-code-review

      cost compare --archs haiku,sonnet,opus --task-type code-review

      cost compare --archs single-agent-haiku,3-agent-sonnet --with-ledger
    """
    from costmodel.comparator import ArchitectureComparator
    from costmodel.models import get_architecture, list_architectures, BUILTIN_ARCHITECTURES
    from costmodel.pricing import MODEL_ALIASES

    # Build architecture list
    arch_names: list[str] = []
    if archs:
        arch_names = [a.strip() for a in archs.split(",")]
    else:
        # Default: compare the three main single-agent architectures
        arch_names = ["single-agent-haiku", "single-agent-sonnet", "single-agent-opus"]
        console.print("[dim]No --archs specified; comparing single-agent haiku/sonnet/opus.[/dim]")

    architectures = []
    for name in arch_names:
        # Try direct lookup first
        arch = get_architecture(name)
        if arch is None:
            # Try interpreting as a model alias → single-agent arch
            model_map = {
                "haiku": "single-agent-haiku",
                "sonnet": "single-agent-sonnet",
                "opus": "single-agent-opus",
            }
            if name in model_map:
                arch = get_architecture(model_map[name])
            else:
                # Try as YAML file
                p = Path(name)
                if p.exists():
                    from costmodel.estimator import load_architecture_from_yaml
                    arch = load_architecture_from_yaml(p)
                else:
                    console.print(f"[red]Error:[/red] Unknown architecture '{name}'.")
                    sys.exit(1)
        architectures.append(arch)

    # Apply task-type overrides
    if task_type:
        from costmodel.pricing import TASK_INPUT_TOKENS, TASK_OUTPUT_TOKENS
        in_tok = TASK_INPUT_TOKENS.get(task_type, 3000)
        out_tok = TASK_OUTPUT_TOKENS.get(task_type, 400)
        for arch in architectures:
            for stage in arch.stages:
                for call in stage.calls:
                    call.input_tokens = in_tok
                    call.output_tokens = out_tok

    ledger = _get_ledger() if with_ledger else None
    comp = ArchitectureComparator(ledger=ledger)
    for arch in architectures:
        comp.add(arch)

    # Set baseline
    if baseline:
        # Check if baseline is in our list
        baseline_names = [a.name for a in architectures]
        if baseline not in baseline_names:
            # Try to add it
            baseline_arch = get_architecture(baseline)
            if baseline_arch is None:
                console.print(f"[red]Error:[/red] Unknown baseline architecture '{baseline}'.")
                sys.exit(1)
            if task_type:
                from costmodel.pricing import TASK_INPUT_TOKENS, TASK_OUTPUT_TOKENS
                in_tok = TASK_INPUT_TOKENS.get(task_type, 3000)
                out_tok = TASK_OUTPUT_TOKENS.get(task_type, 400)
                for stage in baseline_arch.stages:
                    for call in stage.calls:
                        call.input_tokens = in_tok
                        call.output_tokens = out_tok
            comp.add(baseline_arch)
        comp.set_baseline(baseline)

    report = comp.compare()

    # ── Print comparison table ───────────────────────────────────────────────
    title = "Architecture Comparison"
    if task_type:
        title += f": {task_type}"
    console.print()
    console.print(f"[bold]{title}[/bold]")
    console.print()

    tbl = Table(box=box.SIMPLE_HEAVY, show_footer=False)
    tbl.add_column("Architecture", style="cyan")
    tbl.add_column("Est/Run", justify="right", style="yellow")
    tbl.add_column("Act/Run", justify="right")
    tbl.add_column("At 1K/day", justify="right")
    tbl.add_column("Runs", justify="right")
    if report.baseline_name:
        tbl.add_column("vs Baseline", justify="right", style="magenta")

    for row in report.rows:
        act_str = _fmt_usd(row.act_per_run) if row.act_per_run is not None else "[dim]N/A[/dim]"
        baseline_str = ""
        if report.baseline_name:
            if row.name == report.baseline_name:
                baseline_str = "[dim]baseline[/dim]"
            elif row.vs_baseline_ratio is not None:
                ratio = row.vs_baseline_ratio
                if ratio < 1.0:
                    baseline_str = f"[green]{ratio:.4f}×[/green]"
                else:
                    baseline_str = f"[red]{ratio:.2f}×[/red]"

        row_data = [
            row.name,
            _fmt_usd(row.est_per_run),
            act_str,
            _fmt_usd(row.per_1000_day),
            str(row.run_count) if row.run_count > 0 else "[dim]0[/dim]",
        ]
        if report.baseline_name:
            row_data.append(baseline_str)

        tbl.add_row(*row_data)

    console.print(tbl)

    # ── Cherny hypothesis check ──────────────────────────────────────────────
    if report.cherny:
        ch = report.cherny
        console.print()
        console.print("[bold]Cherny Hypothesis Check[/bold]")
        console.print(
            f"  Comparing [cyan]{ch.model_a}[/cyan] (expensive) "
            f"vs [cyan]{ch.model_b}[/cyan] (cheap)"
        )
        console.print(
            f"  Cost ratio: [yellow]{ch.cost_ratio:.1f}×[/yellow] "
            f"({_fmt_usd(ch.cost_a)} vs {_fmt_usd(ch.cost_b)}) — "
            f"[dim]source: {ch.data_source}[/dim]"
        )
        console.print(
            f"  Breakeven: if expensive model needs [yellow]{ch.breakeven_ratio:.1f}×[/yellow] "
            f"fewer retries/corrections, it wins on total cost"
        )

        if ch.verdict == "INSUFFICIENT_DATA":
            console.print(f"  [yellow]⚠ {ch.verdict_detail}[/yellow]")
        elif ch.verdict == "MODEL_A_WINS":
            console.print(f"  [green]✓ {ch.verdict_detail}[/green]")
        elif ch.verdict == "MODEL_B_WINS":
            console.print(f"  [red]✗ {ch.verdict_detail}[/red]")
        else:
            console.print(f"  [yellow]~ {ch.verdict_detail}[/yellow]")

    console.print()


# ---------------------------------------------------------------------------
# cost report
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--last", type=int, default=50, show_default=True, help=(
    "Number of recent runs to include."
))
@click.option("--arch", "architecture_name", default=None, help=(
    "Filter to a specific architecture name."
))
@click.option("--output", "output_path", default=None, help=(
    "Save report to this file (markdown). Prints to stdout if omitted."
))
def report(
    last: int,
    architecture_name: Optional[str],
    output_path: Optional[str],
) -> None:
    """Generate a cost report from recorded runs.

    Examples:

      cost report --last 50 --output report.md

      cost report --arch single-agent-opus --last 20

      cost report --last 100
    """
    from costmodel.reporter import generate_report, save_report

    ledger = _get_ledger()
    md = generate_report(ledger, limit=last, architecture_name=architecture_name)

    if output_path:
        save_report(md, output_path)
        console.print(f"[green]Report saved to[/green] [cyan]{output_path}[/cyan]")
    else:
        console.print(md)


# ---------------------------------------------------------------------------
# cost cherny-check
# ---------------------------------------------------------------------------


@cli.command(name="cherny-check")
@click.option("--model", "model_a", default="opus", show_default=True, help=(
    "The 'better' (more expensive) model to check."
))
@click.option("--vs", "model_b", default="haiku", show_default=True, help=(
    "The cheaper baseline model to compare against."
))
@click.option("--min-runs", type=int, default=10, show_default=True, help=(
    "Minimum real runs required for each model."
))
def cherny_check(
    model_a: str,
    model_b: str,
    min_runs: int,
) -> None:
    """Validate the Cherny hypothesis on real ledger data.

    The Cherny hypothesis: 'Use the best model — it's actually cheaper
    because it requires fewer corrections/retries.'

    Requires ≥10 real recorded runs per model in the ledger.

    Examples:

      cost cherny-check --model opus --vs haiku

      cost cherny-check --model sonnet --vs haiku --min-runs 20
    """
    from costmodel.comparator import ArchitectureComparator
    from costmodel.models import get_architecture, Architecture, Stage, ModelCall
    from costmodel.pricing import resolve_model

    ledger = _get_ledger()

    # Resolve model names to canonical form
    try:
        model_a_canonical = resolve_model(model_a)
        model_b_canonical = resolve_model(model_b)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    # Map model → architecture name in ledger
    # We look for runs where architecture_name contains the model name
    stats_a = ledger.architecture_stats(f"single-agent-{model_a}")
    stats_b = ledger.architecture_stats(f"single-agent-{model_b}")

    console.print()
    console.print("[bold]Cherny Hypothesis Check[/bold]")
    console.print(
        f"  Model A (expensive): [cyan]{model_a_canonical}[/cyan]"
    )
    console.print(
        f"  Model B (cheap):     [cyan]{model_b_canonical}[/cyan]"
    )
    console.print()

    runs_a = stats_a.run_count if stats_a else 0
    runs_b = stats_b.run_count if stats_b else 0

    console.print(f"  Recorded runs — Model A: [yellow]{runs_a}[/yellow], Model B: [yellow]{runs_b}[/yellow]")
    console.print(f"  Required minimum: [yellow]{min_runs}[/yellow] per model")
    console.print()

    if runs_a < min_runs or runs_b < min_runs:
        console.print("[yellow]⚠ Insufficient real data for statistical validation.[/yellow]")
        console.print()

        missing_a = max(0, min_runs - runs_a)
        missing_b = max(0, min_runs - runs_b)
        if missing_a > 0:
            console.print(
                f"  Need [red]{missing_a}[/red] more runs with architecture "
                f"[cyan]single-agent-{model_a}[/cyan]"
            )
        if missing_b > 0:
            console.print(
                f"  Need [red]{missing_b}[/red] more runs with architecture "
                f"[cyan]single-agent-{model_b}[/cyan]"
            )
        console.print()
        console.print("[dim]Showing estimate-only comparison instead:[/dim]")
        console.print()

        # Fall back to estimate-only comparison
        arch_a = get_architecture(f"single-agent-{model_a}") or Architecture(
            name=f"single-agent-{model_a}",
            stages=[Stage("main", [ModelCall(model_a_canonical, 3000, 400)])],
        )
        arch_b = get_architecture(f"single-agent-{model_b}") or Architecture(
            name=f"single-agent-{model_b}",
            stages=[Stage("main", [ModelCall(model_b_canonical, 3000, 400)])],
        )

        from costmodel.estimator import estimate as do_estimate
        est_a = do_estimate(arch_a)
        est_b = do_estimate(arch_b)
        ratio = est_a.per_run_usd / est_b.per_run_usd if est_b.per_run_usd > 0 else float("inf")

        tbl = Table(box=box.ROUNDED)
        tbl.add_column("Model", style="cyan")
        tbl.add_column("Est/Run", justify="right", style="yellow")
        tbl.add_column("Per 1K runs", justify="right")
        tbl.add_column("Recorded Runs", justify="right")

        tbl.add_row(
            model_a_canonical, _fmt_usd(est_a.per_run_usd),
            _fmt_usd(est_a.per_1000_runs_usd), str(runs_a)
        )
        tbl.add_row(
            model_b_canonical, _fmt_usd(est_b.per_run_usd),
            _fmt_usd(est_b.per_1000_runs_usd), str(runs_b)
        )
        console.print(tbl)
        console.print()
        console.print(
            f"  [yellow]{model_a_canonical}[/yellow] is [yellow]{ratio:.1f}×[/yellow] more expensive per call (estimated)."
        )
        console.print(
            f"  Cherny hypothesis holds if {model_a_canonical} needs "
            f"[yellow]{ratio:.1f}×[/yellow] fewer corrections than {model_b_canonical}."
        )
        console.print()
        console.print(
            f"  Record real runs with [cyan]cost track --arch single-agent-{model_a}[/cyan] "
            f"to validate empirically."
        )
        console.print()
        return

    # We have enough data — show real comparison
    avg_a = stats_a.avg_cost_usd  # type: ignore[union-attr]
    avg_b = stats_b.avg_cost_usd  # type: ignore[union-attr]
    ratio = avg_a / avg_b if avg_b > 0 else float("inf")

    tbl = Table(box=box.ROUNDED, title="Real Run Data")
    tbl.add_column("Model", style="cyan")
    tbl.add_column("Runs", justify="right")
    tbl.add_column("Avg/Run", justify="right", style="yellow")
    tbl.add_column("p50/Run", justify="right")
    tbl.add_column("p95/Run", justify="right")
    tbl.add_column("Total Spent", justify="right")

    tbl.add_row(
        model_a_canonical,
        str(stats_a.run_count),  # type: ignore[union-attr]
        _fmt_usd(stats_a.avg_cost_usd),  # type: ignore[union-attr]
        _fmt_usd(stats_a.p50_cost_usd),  # type: ignore[union-attr]
        _fmt_usd(stats_a.p95_cost_usd),  # type: ignore[union-attr]
        _fmt_usd(stats_a.total_cost_usd),  # type: ignore[union-attr]
    )
    tbl.add_row(
        model_b_canonical,
        str(stats_b.run_count),  # type: ignore[union-attr]
        _fmt_usd(stats_b.avg_cost_usd),  # type: ignore[union-attr]
        _fmt_usd(stats_b.p50_cost_usd),  # type: ignore[union-attr]
        _fmt_usd(stats_b.p95_cost_usd),  # type: ignore[union-attr]
        _fmt_usd(stats_b.total_cost_usd),  # type: ignore[union-attr]
    )
    console.print(tbl)
    console.print()

    console.print(f"  Cost ratio: [yellow]{ratio:.1f}×[/yellow] ({model_a} vs {model_b})")

    if avg_a <= avg_b:
        console.print(
            f"  [green]✓ CHERNY HYPOTHESIS CONFIRMED[/green] — "
            f"{model_a_canonical} costs less per run despite higher per-token price."
        )
    elif ratio < 2.0:
        console.print(
            f"  [yellow]~ BORDERLINE[/yellow] — only {ratio:.1f}× difference. "
            f"Depends on your specific task distribution."
        )
    else:
        console.print(
            f"  [red]✗ CHERNY HYPOTHESIS NOT CONFIRMED[/red] — "
            f"{model_b_canonical} is cheaper at avg {_fmt_usd(avg_b)}/run vs "
            f"{_fmt_usd(avg_a)}/run for {model_a_canonical}."
        )
        console.print(
            f"  {model_a_canonical} would need to reduce retries by {ratio:.1f}× to break even."
        )
    console.print()


# ---------------------------------------------------------------------------
# cost track
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--run-id", required=True, help="Unique identifier for this run.")
@click.option("--arch", "arch_name", required=True, help="Architecture name for this run.")
@click.option("--model", required=True, help="Model used in this call.")
@click.option("--input-tokens", type=int, required=True, help="Input tokens used.")
@click.option("--output-tokens", type=int, required=True, help="Output tokens used.")
@click.option("--cached-tokens", type=int, default=0, help="Cached input tokens.")
@click.option("--stage", default="main", help="Stage name within the architecture.")
@click.option("--latency-ms", type=int, default=None, help="Latency in milliseconds.")
@click.option("--finish", is_flag=True, default=False, help="Mark this run as complete.")
def track(
    run_id: str,
    arch_name: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    stage: str,
    latency_ms: Optional[int],
    finish: bool,
) -> None:
    """Record a real API call to the cost ledger.

    Examples:

      cost track --run-id abc123 --arch single-agent-opus \\
                 --model opus --input-tokens 3000 --output-tokens 400

      cost track --run-id abc123 --arch single-agent-opus \\
                 --model opus --input-tokens 3000 --output-tokens 400 --finish
    """
    from costmodel.pricing import cost_for_call, resolve_model

    try:
        canonical_model = resolve_model(model)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    call_cost = cost_for_call(
        model=canonical_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_tokens,
    )

    ledger = _get_ledger()
    row_id = ledger.record_call(
        run_id=run_id,
        model=canonical_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=call_cost,
        architecture_name=arch_name,
        stage_name=stage,
        cached_input_tokens=cached_tokens,
        latency_ms=latency_ms,
    )

    console.print(
        f"[green]Recorded[/green] call #{row_id} — "
        f"run=[cyan]{run_id}[/cyan] model=[cyan]{canonical_model}[/cyan] "
        f"cost=[yellow]{_fmt_usd(call_cost)}[/yellow]"
    )

    if finish:
        summary = ledger.finish_run(run_id=run_id, architecture_name=arch_name)
        console.print(
            f"[green]Run complete[/green] — "
            f"total cost: [yellow]{_fmt_usd(summary.total_cost_usd)}[/yellow] "
            f"tokens: {_fmt_tokens(summary.total_tokens)}"
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entrypoint."""
    cli()
