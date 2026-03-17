"""HighlightCompiler - assemble clips into a highlight reel with transitions."""

from __future__ import annotations

from highlightai.models import GameFootage, Highlight, HighlightReel


class HighlightCompiler:
    """Assemble ordered Highlight clips into a HighlightReel.

    In production this would render actual video; here it builds the metadata
    structure and assigns transitions.
    """

    TRANSITIONS = ["cut", "fade", "dissolve", "wipe", "zoom"]

    def __init__(
        self,
        default_transition: str = "fade",
        intro_seconds: float = 2.0,
        outro_seconds: float = 3.0,
    ) -> None:
        self.default_transition = default_transition
        self.intro_seconds = intro_seconds
        self.outro_seconds = outro_seconds

    def compile(
        self,
        highlights: list[Highlight],
        footage: GameFootage,
    ) -> HighlightReel:
        """Build a HighlightReel from ordered highlights."""
        # Assign transitions
        processed = self._assign_transitions(highlights)

        reel = HighlightReel(
            footage=footage,
            highlights=processed,
        )
        reel.compute_stats()
        # Add intro/outro time
        reel.total_duration += self.intro_seconds + self.outro_seconds
        return reel

    def _assign_transitions(
        self,
        highlights: list[Highlight],
    ) -> list[Highlight]:
        """Assign transition types between clips based on temporal distance."""
        if not highlights:
            return []

        result: list[Highlight] = []
        for i, h in enumerate(highlights):
            transition_in = self.default_transition
            transition_out = self.default_transition

            if i == 0:
                transition_in = "fade"  # intro always fades in

            if i == len(highlights) - 1:
                transition_out = "fade"  # outro always fades out
            elif i < len(highlights) - 1:
                gap = highlights[i + 1].clip_start - h.clip_end
                if gap < 2.0:
                    transition_out = "cut"  # quick cut for close clips
                elif gap < 10.0:
                    transition_out = "dissolve"
                else:
                    transition_out = "fade"

            result.append(
                Highlight(
                    moment=h.moment,
                    clip_start=h.clip_start,
                    clip_end=h.clip_end,
                    rank=h.rank,
                    transition_in=transition_in,
                    transition_out=transition_out,
                )
            )

        return result

    def reorder_chronologically(
        self,
        highlights: list[Highlight],
    ) -> list[Highlight]:
        """Reorder highlights by timestamp (chronological order)."""
        sorted_hl = sorted(highlights, key=lambda h: h.moment.timestamp)
        for i, h in enumerate(sorted_hl):
            h.rank = i + 1
        return sorted_hl

    def estimate_render_time(self, reel: HighlightReel) -> float:
        """Estimate rendering time in seconds (rough heuristic)."""
        # Assume 2x realtime rendering
        return reel.total_duration * 2.0
