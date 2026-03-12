"""CLI entrypoint for the AI Cost Modeler."""

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option()
def cli() -> None:
    """AI Cost Modeler — estimate and track AI inference costs.

    Model architecture costs before you build, track actuals while you run,
    and validate the 'use the best model' hypothesis with real data.
    """


@cli.command()
@click.argument("architecture", required=False)
def estimate(architecture: str | None) -> None:
    """Estimate cost for a given architecture description."""
    console.print("[yellow]estimate[/yellow]: not yet implemented (see issue #7)")


@cli.command()
@click.argument("arch_a", required=False)
@click.argument("arch_b", required=False)
def compare(arch_a: str | None, arch_b: str | None) -> None:
    """Compare two architectures side-by-side on cost."""
    console.print("[yellow]compare[/yellow]: not yet implemented (see issue #4)")


@cli.command()
def report() -> None:
    """Show cost report: estimates vs actual spend."""
    console.print("[yellow]report[/yellow]: not yet implemented (see issue #2)")


@cli.command(name="cherny-check")
def cherny_check() -> None:
    """Run 20 tasks with Haiku vs Opus and compare total cost."""
    console.print("[yellow]cherny-check[/yellow]: not yet implemented (see issue #3)")


def main() -> None:
    """Main entrypoint."""
    cli()
