"""Tests for ExcitementDetector."""

import numpy as np
import pytest

from highlightai.detector.excitement import ExcitementDetector
from highlightai.models import AudioFeatures, MotionFeatures


class TestExcitementDetector:
    def setup_method(self):
        self.detector = ExcitementDetector(threshold=0.5)

    def test_score_frame_zeros(self):
        af = AudioFeatures(timestamp=0.0)
        mf = MotionFeatures(timestamp=0.0)
        score = self.detector.score_frame(af, mf)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_score_frame_max(self):
        af = AudioFeatures(timestamp=0.0, crowd_noise_level=1.0, commentary_excitement=1.0)
        mf = MotionFeatures(timestamp=0.0, motion_intensity=1.0, celebration_probability=1.0)
        score = self.detector.score_frame(af, mf)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_detect_exciting_moments(self):
        # Build features with a clear spike
        audio = []
        for i in range(100):
            ts = float(i)
            crowd = 0.9 if 40 <= i <= 45 else 0.1
            audio.append(AudioFeatures(
                timestamp=ts,
                crowd_noise_level=crowd,
                commentary_excitement=crowd * 0.8,
            ))
        motion = [
            MotionFeatures(
                timestamp=float(i),
                motion_intensity=0.8 if 40 <= i <= 45 else 0.1,
            )
            for i in range(100)
        ]

        moments = self.detector.detect_exciting_moments(audio, motion, min_gap_seconds=5.0)
        assert len(moments) >= 1
        # The moment should be around t=40
        assert any(35 <= m.timestamp <= 50 for m in moments)

    def test_no_motion_features(self):
        audio = [
            AudioFeatures(timestamp=float(i), crowd_noise_level=0.9)
            for i in range(10)
        ]
        moments = self.detector.detect_exciting_moments(audio, None)
        assert isinstance(moments, list)

    def test_compute_motion_features(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        prev = np.ones((100, 100, 3), dtype=np.uint8) * 128
        mf = self.detector.compute_motion_features(frame, prev, timestamp=1.0)
        assert mf.motion_intensity > 0
        assert mf.timestamp == 1.0
