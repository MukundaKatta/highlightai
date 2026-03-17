"""Tests for EventDetector."""

from highlightai.detector.events import EventDetector
from highlightai.models import AudioFeatures, EventType, MotionFeatures, Sport


class TestEventDetector:
    def test_detect_soccer_goal(self):
        detector = EventDetector(Sport.SOCCER)
        audio = [
            AudioFeatures(
                timestamp=10.0,
                crowd_noise_level=0.9,
                commentary_excitement=0.9,
            )
        ]
        motion = [
            MotionFeatures(timestamp=10.0, motion_intensity=0.7)
        ]
        events = detector.detect_events(audio, motion)
        assert len(events) >= 1
        assert events[0].event_type == EventType.GOAL

    def test_detect_basketball_dunk(self):
        detector = EventDetector(Sport.BASKETBALL)
        audio = [
            AudioFeatures(
                timestamp=5.0,
                crowd_noise_level=0.85,
                commentary_excitement=0.9,
            )
        ]
        motion = [
            MotionFeatures(timestamp=5.0, motion_intensity=0.8)
        ]
        events = detector.detect_events(audio, motion)
        assert len(events) >= 1
        assert events[0].event_type == EventType.DUNK

    def test_no_event_when_quiet(self):
        detector = EventDetector(Sport.SOCCER)
        audio = [
            AudioFeatures(timestamp=1.0, crowd_noise_level=0.1, commentary_excitement=0.1)
        ]
        motion = [MotionFeatures(timestamp=1.0, motion_intensity=0.1)]
        events = detector.detect_events(audio, motion)
        assert len(events) == 0

    def test_supported_events(self):
        for sport in Sport:
            detector = EventDetector(sport)
            events = detector.supported_events()
            assert len(events) > 0

    def test_all_sports(self):
        sports = EventDetector.all_sports()
        assert len(sports) == len(Sport)

    def test_events_for_sport(self):
        events = EventDetector.events_for_sport(Sport.CRICKET)
        assert any(e == EventType.WICKET for e, _ in events)

    def test_deduplication_min_gap(self):
        detector = EventDetector(Sport.SOCCER)
        # Two events very close together
        audio = [
            AudioFeatures(timestamp=float(i), crowd_noise_level=0.9, commentary_excitement=0.9)
            for i in range(5)
        ]
        motion = [
            MotionFeatures(timestamp=float(i), motion_intensity=0.7)
            for i in range(5)
        ]
        events = detector.detect_events(audio, motion, min_gap_seconds=10.0)
        # Should be deduplicated to 1
        assert len(events) == 1
