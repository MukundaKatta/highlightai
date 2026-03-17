"""Tests for HighlightAI pydantic models."""

import pytest

from highlightai.models import (
    AudioFeatures,
    EventType,
    GameFootage,
    Highlight,
    HighlightReel,
    Moment,
    MotionFeatures,
    Sport,
)


class TestAudioFeatures:
    def test_defaults(self):
        af = AudioFeatures(timestamp=10.0)
        assert af.crowd_noise_level == 0.0
        assert af.is_crowd_roar is False

    def test_bounds(self):
        with pytest.raises(Exception):
            AudioFeatures(timestamp=0.0, crowd_noise_level=1.5)


class TestMotionFeatures:
    def test_defaults(self):
        mf = MotionFeatures()
        assert mf.motion_intensity == 0.0

    def test_create(self):
        mf = MotionFeatures(
            timestamp=5.0,
            motion_intensity=0.8,
            celebration_probability=0.6,
        )
        assert mf.motion_intensity == 0.8


class TestGameFootage:
    def test_create(self):
        gf = GameFootage(
            sport=Sport.SOCCER,
            duration_seconds=5400,
            home_team="Eagles",
            away_team="Lions",
        )
        assert gf.fps == 30.0
        assert gf.resolution == (1920, 1080)


class TestMoment:
    def test_create(self):
        m = Moment(
            timestamp=120.0,
            event_type=EventType.GOAL,
            excitement_score=0.95,
            description="Header from corner",
        )
        assert m.event_type == EventType.GOAL

    def test_no_event_type(self):
        m = Moment(timestamp=50.0, excitement_score=0.6)
        assert m.event_type is None


class TestHighlight:
    def test_create(self):
        m = Moment(timestamp=100.0, excitement_score=0.8)
        h = Highlight(moment=m, clip_start=97.0, clip_end=108.0, rank=1)
        assert h.clip_end - h.clip_start == 11.0


class TestHighlightReel:
    def test_compute_stats(self):
        m1 = Moment(timestamp=100.0, excitement_score=0.8)
        m2 = Moment(timestamp=200.0, excitement_score=0.6)
        h1 = Highlight(moment=m1, clip_start=97.0, clip_end=108.0)
        h2 = Highlight(moment=m2, clip_start=197.0, clip_end=210.0)
        footage = GameFootage(duration_seconds=5400)
        reel = HighlightReel(footage=footage, highlights=[h1, h2])
        reel.compute_stats()
        assert reel.num_moments == 2
        assert reel.total_duration == pytest.approx(24.0)
        assert reel.average_excitement == pytest.approx(0.7)
