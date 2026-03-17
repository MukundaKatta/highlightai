"""Tests for HighlightClipper."""

import pytest

from highlightai.editor.clipper import HighlightClipper
from highlightai.models import EventType, GameFootage, Moment


class TestHighlightClipper:
    def setup_method(self):
        self.clipper = HighlightClipper(pre_context=3.0, post_context=5.0)
        self.footage = GameFootage(duration_seconds=5400)

    def test_clip_moments(self):
        moments = [
            Moment(timestamp=100.0, duration=5.0, excitement_score=0.9),
            Moment(timestamp=500.0, duration=5.0, excitement_score=0.7),
        ]
        clips = self.clipper.clip_moments(moments, self.footage)
        assert len(clips) == 2
        assert clips[0].clip_start == 97.0
        assert clips[0].clip_end == 110.0
        assert clips[0].rank == 1

    def test_top_n(self):
        moments = [
            Moment(timestamp=float(i * 100), excitement_score=0.8)
            for i in range(20)
        ]
        clips = self.clipper.clip_moments(moments, self.footage, top_n=5)
        assert len(clips) == 5

    def test_clamp_to_footage_bounds(self):
        moments = [
            Moment(timestamp=1.0, duration=5.0, excitement_score=0.9),
            Moment(timestamp=5395.0, duration=10.0, excitement_score=0.8),
        ]
        clips = self.clipper.clip_moments(moments, self.footage)
        assert clips[0].clip_start >= 0.0
        assert clips[1].clip_end <= 5400.0

    def test_merge_overlapping(self):
        moments = [
            Moment(timestamp=100.0, duration=5.0, excitement_score=0.9),
            Moment(timestamp=103.0, duration=5.0, excitement_score=0.7),  # overlaps
        ]
        clips = self.clipper.clip_moments(moments, self.footage)
        merged = self.clipper.merge_overlapping(clips)
        assert len(merged) == 1
        assert merged[0].rank == 1

    def test_no_merge_when_far_apart(self):
        moments = [
            Moment(timestamp=100.0, duration=5.0, excitement_score=0.9),
            Moment(timestamp=500.0, duration=5.0, excitement_score=0.7),
        ]
        clips = self.clipper.clip_moments(moments, self.footage)
        merged = self.clipper.merge_overlapping(clips)
        assert len(merged) == 2

    def test_total_duration(self):
        moments = [
            Moment(timestamp=100.0, duration=5.0, excitement_score=0.9),
        ]
        clips = self.clipper.clip_moments(moments, self.footage)
        total = self.clipper.total_duration(clips)
        assert total == pytest.approx(13.0)  # 3 pre + 5 + 5 post
