"""Report - rich console reports for highlight reels."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from highlightai.models import HighlightReel


def _format_time(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _excitement_colour(score: float) -> str:
    if score >= 0.8:
        return "bold red"
    if score >= 0.6:
        return "yellow"
    return "dim"


def render_highlight_reel(
    reel: HighlightReel,
    console: Console | None = None,
) -> None:
    """Print a rich summary of the highlight reel."""
    console = console or Console()

    # Header
    console.print()
    sport_label = reel.footage.sport.value.upper()
    teams = ""
    if reel.footage.home_team and reel.footage.away_team:
        teams = f"  {reel.footage.home_team} vs {reel.footage.away_team}"
    console.print(
        Panel(
            f"[bold]{sport_label} HIGHLIGHTS[/bold]{teams}\n"
            f"Game duration: {_format_time(reel.footage.duration_seconds)}  |  "
            f"Clips: {reel.num_moments}  |  "
            f"Reel length: {_format_time(reel.total_duration)}  |  "
            f"Avg excitement: {reel.average_excitement:.2f}",
            title="Highlight Reel",
            border_style="cyan",
        )
    )

    # Clips table
    table = Table(title="Highlights", show_lines=True)
    table.add_column("#", style="bold", justify="center", width=4)
    table.add_column("Time", justify="center", width=12)
    table.add_column("Event", width=18)
    table.add_column("Description", width=35)
    table.add_column("Excitement", justify="center", width=12)
    table.add_column("Clip Range", justify="center", width=14)
    table.add_column("Transitions", style="dim", width=16)

    for h in reel.highlights:
        m = h.moment
        colour = _excitement_colour(m.excitement_score)
        event_name = m.event_type.value.replace("_", " ").title() if m.event_type else "Unknown"
        table.add_row(
            str(h.rank),
            _format_time(m.timestamp),
            event_name,
            m.description or "-",
            Text(f"{m.excitement_score:.2f}", style=colour),
            f"{_format_time(h.clip_start)}-{_format_time(h.clip_end)}",
            f"{h.transition_in} / {h.transition_out}",
        )

    console.print(table)

    # Summary
    console.print()
    if reel.average_excitement >= 0.8:
        console.print(
            "[bold red]An absolute thriller! This reel has it all.[/bold red]"
        )
    elif reel.average_excitement >= 0.6:
        console.print(
            "[bold yellow]A solid set of highlights with plenty of action.[/bold yellow]"
        )
    else:
        console.print("[dim]A quieter game, but still some moments worth watching.[/dim]")
    console.print()
