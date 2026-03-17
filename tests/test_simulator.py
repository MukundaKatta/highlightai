"""Tests for GameSimulator."""

from highlightai.models import Sport
from highlightai.simulator import GameSimulator


class TestGameSimulator:
    def test_footage_metadata(self):
        sim = GameSimulator(Sport.SOCCER, duration_minutes=90.0, seed=0)
        footage = sim.footage_metadata()
        assert footage.sport == Sport.SOCCER
        assert footage.duration_seconds == 5400.0

    def test_generate_audio_features(self):
        sim = GameSimulator(Sport.SOCCER, duration_minutes=10.0, seed=1)
        features = sim.generate_audio_features(window_seconds=1.0)
        assert len(features) == 600  # 10 min * 60 sec / 1 sec window
        for f in features:
            assert 0 <= f.crowd_noise_level <= 1.0
            assert 0 <= f.commentary_excitement <= 1.0

    def test_generate_motion_features(self):
        sim = GameSimulator(Sport.BASKETBALL, duration_minutes=5.0, seed=2)
        features = sim.generate_motion_features(window_seconds=1.0)
        assert len(features) == 300
        for f in features:
            assert 0 <= f.motion_intensity <= 1.0

    def test_generate_expected_moments(self):
        sim = GameSimulator(Sport.SOCCER, seed=3)
        moments = sim.generate_expected_moments()
        assert len(moments) > 0
        for m in moments:
            assert m.event_type is not None
            assert 0 <= m.timestamp <= sim.duration_seconds

    def test_different_sports(self):
        for sport in Sport:
            sim = GameSimulator(sport, duration_minutes=10.0, seed=10)
            audio = sim.generate_audio_features()
            motion = sim.generate_motion_features()
            assert len(audio) > 0
            assert len(motion) > 0

    def test_audio_has_spikes_near_events(self):
        sim = GameSimulator(Sport.SOCCER, duration_minutes=90.0, seed=5)
        audio = sim.generate_audio_features()
        moments = sim.generate_expected_moments()
        # Check that audio near the first event is louder than baseline
        if moments:
            event_ts = moments[0].timestamp
            near_event = [
                f for f in audio if abs(f.timestamp - event_ts) < 5.0
            ]
            far_from_event = [
                f for f in audio
                if all(abs(f.timestamp - m.timestamp) > 30 for m in moments)
            ]
            if near_event and far_from_event:
                avg_near = sum(f.crowd_noise_level for f in near_event) / len(near_event)
                avg_far = sum(f.crowd_noise_level for f in far_from_event) / len(far_from_event)
                assert avg_near > avg_far
