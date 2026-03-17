"""Tests for HighlightCompiler."""

import pytest

from highlightai.editor.compiler import HighlightCompiler
from highlightai.models import GameFootage, Highlight, Moment


class TestHighlightCompiler:
    def setup_method(self):
        self.compiler = HighlightCompiler()
        self.footage = GameFootage(duration_seconds=5400)

    def _make_highlights(self) -> list[Highlight]:
        return [
            Highlight(
                moment=Moment(timestamp=100.0, excitement_score=0.9, description="Goal"),
                clip_start=97.0,
                clip_end=110.0,
                rank=1,
            ),
            Highlight(
                moment=Moment(timestamp=500.0, excitement_score=0.7, description="Save"),
                clip_start=497.0,
                clip_end=510.0,
                rank=2,
            ),
            Highlight(
                moment=Moment(timestamp=2000.0, excitement_score=0.8, description="Penalty"),
                clip_start=1997.0,
                clip_end=2013.0,
                rank=3,
            ),
        ]

    def test_compile_reel(self):
        highlights = self._make_highlights()
        reel = self.compiler.compile(highlights, self.footage)
        assert reel.num_moments == 3
        assert reel.total_duration > 0
        assert reel.average_excitement > 0

    def test_transitions_assigned(self):
        highlights = self._make_highlights()
        reel = self.compiler.compile(highlights, self.footage)
        # First clip fades in
        assert reel.highlights[0].transition_in == "fade"
        # Last clip fades out
        assert reel.highlights[-1].transition_out == "fade"

    def test_reorder_chronologically(self):
        highlights = [
            Highlight(
                moment=Moment(timestamp=500.0, excitement_score=0.9),
                clip_start=497.0, clip_end=510.0, rank=1,
            ),
            Highlight(
                moment=Moment(timestamp=100.0, excitement_score=0.7),
                clip_start=97.0, clip_end=110.0, rank=2,
            ),
        ]
        reordered = self.compiler.reorder_chronologically(highlights)
        assert reordered[0].moment.timestamp < reordered[1].moment.timestamp
        assert reordered[0].rank == 1

    def test_estimate_render_time(self):
        highlights = self._make_highlights()
        reel = self.compiler.compile(highlights, self.footage)
        render_time = self.compiler.estimate_render_time(reel)
        assert render_time == pytest.approx(reel.total_duration * 2.0)

    def test_empty_highlights(self):
        reel = self.compiler.compile([], self.footage)
        assert reel.num_moments == 0
        # Only intro + outro time
        assert reel.total_duration == pytest.approx(5.0)
