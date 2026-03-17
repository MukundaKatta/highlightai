"""CLI entry-point for HighlightAI."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from highlightai.detector.events import EventDetector
from highlightai.detector.excitement import ExcitementDetector
from highlightai.editor.clipper import HighlightClipper
from highlightai.editor.compiler import HighlightCompiler
from highlightai.editor.ranker import MomentRanker
from highlightai.models import Sport
from highlightai.report import render_highlight_reel
from highlightai.simulator import GameSimulator

console = Console()


@click.group()
@click.version_option(package_name="highlightai")
def cli() -> None:
    """HighlightAI - Sports Highlight Generator."""


@cli.command()
def sports() -> None:
    """List supported sports and their event types."""
    table = Table(title="Supported Sports & Events", show_lines=True)
    table.add_column("Sport", style="bold")
    table.add_column("Events")

    for sport in Sport:
        events = EventDetector.events_for_sport(sport)
        event_list = ", ".join(f"{e.value} ({d})" for e, d in events)
        table.add_row(sport.value.title(), event_list or "None defined")

    console.print(table)


@cli.command()
@click.option(
    "--sport",
    "-s",
    type=click.Choice([s.value for s in Sport], case_sensitive=False),
    default="soccer",
    help="Sport type.",
)
@click.option("--duration", "-d", default=90.0, help="Game duration in minutes.")
@click.option("--top", "-n", default=10, help="Number of highlights to include.")
@click.option("--seed", default=None, type=int, help="Random seed.")
def simulate(sport: str, duration: float, top: int, seed: int | None) -> None:
    """Simulate a game and generate a highlight reel report."""
    sport_enum = Sport(sport)
    console.print(
        f"Simulating {duration:.0f}-minute {sport_enum.value} game..."
    )

    sim = GameSimulator(
        sport=sport_enum,
        duration_minutes=duration,
        home_team="Eagles",
        away_team="Lions",
        seed=seed,
    )
    footage = sim.footage_metadata()
    audio = sim.generate_audio_features()
    motion = sim.generate_motion_features()

    # Detect events & excitement
    event_detector = EventDetector(sport_enum)
    excitement_detector = ExcitementDetector()
    events = event_detector.detect_events(audio, motion)
    exciting = excitement_detector.detect_exciting_moments(audio, motion)

    # Merge all moments, preferring events
    all_moments = events + [m for m in exciting if not m.event_type]

    # Rank and select top N
    ranker = MomentRanker(sport_enum)
    ranked = ranker.top_n(all_moments, top)

    # Clip and compile
    clipper = HighlightClipper()
    highlights = clipper.clip_moments(ranked, footage, top_n=top)
    highlights = clipper.merge_overlapping(highlights)

    compiler = HighlightCompiler()
    reel = compiler.compile(highlights, footage)

    render_highlight_reel(reel, console)


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Path to video file.")
@click.option("--top", "-n", default=10, help="Number of highlights.")
def generate(input_path: str, top: int) -> None:
    """Generate highlights from a game recording (placeholder)."""
    console.print(
        f"[yellow]Video processing for {input_path} is not yet implemented. "
        f"Use 'simulate' for a demo.[/yellow]"
    )


if __name__ == "__main__":
    cli()
