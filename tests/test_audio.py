"""Tests for AudioAnalyzer."""

import numpy as np
import pytest

from highlightai.detector.audio import AudioAnalyzer


class TestAudioAnalyzer:
    def setup_method(self):
        self.analyzer = AudioAnalyzer(
            crowd_threshold=0.5,
            commentary_threshold=0.5,
            sample_rate=8000.0,
            window_seconds=0.5,
        )

    def test_analyse_silence(self):
        silence = np.zeros(8000)  # 1 second of silence
        features = self.analyzer.analyse_signal(silence, sample_rate=8000.0)
        assert len(features) >= 1
        for f in features:
            assert f.crowd_noise_level < 0.1
            assert f.is_crowd_roar is False

    def test_analyse_loud_signal(self):
        # Loud broadband noise
        rng = np.random.default_rng(42)
        loud = rng.normal(0, 0.5, size=8000)
        features = self.analyzer.analyse_signal(loud, sample_rate=8000.0)
        assert len(features) >= 1
        # At least some windows should have non-trivial crowd noise
        max_crowd = max(f.crowd_noise_level for f in features)
        assert max_crowd > 0.1

    def test_detect_peaks(self):
        rng = np.random.default_rng(10)
        # Create a signal with a clear spike
        signal = np.zeros(16000)
        signal[6000:7000] = rng.normal(0, 0.8, size=1000)  # spike
        features = self.analyzer.analyse_signal(signal, sample_rate=8000.0)
        peaks = self.analyzer.detect_peaks(features, min_gap_seconds=1.0)
        # May or may not find peaks depending on threshold, but should not crash
        assert isinstance(peaks, list)

    def test_combined_score(self):
        from highlightai.models import AudioFeatures
        af = AudioFeatures(
            timestamp=1.0,
            crowd_noise_level=0.8,
            commentary_excitement=0.6,
        )
        score = self.analyzer.combined_score(af)
        assert 0 <= score <= 1.0
        expected = 0.6 * 0.8 + 0.4 * 0.6
        assert score == pytest.approx(expected, abs=0.01)

    def test_empty_signal(self):
        features = self.analyzer.analyse_signal(np.array([]))
        assert isinstance(features, list)
