"""HighlightClipper - extract top N moments with surrounding context."""

from __future__ import annotations

from highlightai.models import GameFootage, Highlight, Moment


class HighlightClipper:
    """Extract highlight clips from detected moments.

    Each moment is expanded by a configurable context window so viewers
    can see the build-up and aftermath.
    """

    def __init__(
        self,
        pre_context: float = 3.0,
        post_context: float = 5.0,
    ) -> None:
        """
        Parameters
        ----------
        pre_context:
            Seconds of footage to include before the moment.
        post_context:
            Seconds of footage to include after the moment ends.
        """
        self.pre_context = pre_context
        self.post_context = post_context

    def clip_moments(
        self,
        moments: list[Moment],
        footage: GameFootage,
        top_n: int | None = None,
    ) -> list[Highlight]:
        """Convert moments into highlight clips with context windows.

        Parameters
        ----------
        moments:
            Detected moments, assumed sorted by excitement descending.
        footage:
            Source footage metadata (used to clamp timestamps).
        top_n:
            If provided, only keep the top N moments.
        """
        selected = moments[:top_n] if top_n else moments

        highlights: list[Highlight] = []
        for i, moment in enumerate(selected):
            clip_start = max(0.0, moment.timestamp - self.pre_context)
            clip_end = min(
                footage.duration_seconds,
                moment.timestamp + moment.duration + self.post_context,
            )
            highlights.append(
                Highlight(
                    moment=moment,
                    clip_start=round(clip_start, 2),
                    clip_end=round(clip_end, 2),
                    rank=i + 1,
                )
            )

        return highlights

    def merge_overlapping(
        self,
        highlights: list[Highlight],
    ) -> list[Highlight]:
        """Merge clips that overlap in time to avoid repetition."""
        if not highlights:
            return []

        sorted_hl = sorted(highlights, key=lambda h: h.clip_start)
        merged: list[Highlight] = [sorted_hl[0]]

        for h in sorted_hl[1:]:
            prev = merged[-1]
            if h.clip_start <= prev.clip_end:
                # Merge: extend previous clip, keep higher excitement moment
                best_moment = (
                    h.moment
                    if h.moment.excitement_score > prev.moment.excitement_score
                    else prev.moment
                )
                merged[-1] = Highlight(
                    moment=best_moment,
                    clip_start=prev.clip_start,
                    clip_end=max(prev.clip_end, h.clip_end),
                    rank=prev.rank,
                )
            else:
                merged.append(h)

        # Re-number ranks
        for i, h in enumerate(merged):
            h.rank = i + 1

        return merged

    def total_duration(self, highlights: list[Highlight]) -> float:
        """Sum the durations of all clips."""
        return sum(h.clip_end - h.clip_start for h in highlights)
