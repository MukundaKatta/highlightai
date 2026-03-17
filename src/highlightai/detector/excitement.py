"""ExcitementDetector - score frames by crowd noise, motion, and celebrations."""

from __future__ import annotations

from typing import Optional

import numpy as np

from highlightai.models import AudioFeatures, Moment, MotionFeatures


class ExcitementDetector:
    """Fuse audio, motion, and celebration signals into an excitement score.

    The detector combines:
    - Crowd noise level (from AudioAnalyzer)
    - Motion intensity (optical-flow proxy)
    - Celebration probability (pose / gesture detection proxy)

    Weights are tuneable per-sport.
    """

    def __init__(
        self,
        audio_weight: float = 0.4,
        motion_weight: float = 0.3,
        celebration_weight: float = 0.3,
        threshold: float = 0.6,
    ) -> None:
        assert abs(audio_weight + motion_weight + celebration_weight - 1.0) < 0.01
        self.audio_weight = audio_weight
        self.motion_weight = motion_weight
        self.celebration_weight = celebration_weight
        self.threshold = threshold

    def score_frame(
        self,
        audio: AudioFeatures | None = None,
        motion: MotionFeatures | None = None,
    ) -> float:
        """Compute a single-frame excitement score in [0, 1]."""
        crowd = audio.crowd_noise_level if audio else 0.0
        commentary = audio.commentary_excitement if audio else 0.0
        audio_score = max(crowd, commentary)

        motion_score = motion.motion_intensity if motion else 0.0
        celebration = motion.celebration_probability if motion else 0.0

        score = (
            self.audio_weight * audio_score
            + self.motion_weight * motion_score
            + self.celebration_weight * celebration
        )
        return min(1.0, max(0.0, score))

    def detect_exciting_moments(
        self,
        audio_features: list[AudioFeatures],
        motion_features: list[MotionFeatures] | None = None,
        min_gap_seconds: float = 10.0,
    ) -> list[Moment]:
        """Scan time-series features and return moments above threshold.

        Adjacent exciting frames are merged into a single moment. Moments
        closer than *min_gap_seconds* are merged.
        """
        if motion_features is None:
            motion_features = [MotionFeatures(timestamp=a.timestamp) for a in audio_features]

        # Align by index (assume same temporal sampling)
        n = min(len(audio_features), len(motion_features))
        scores: list[tuple[float, float]] = []  # (timestamp, score)
        for i in range(n):
            s = self.score_frame(audio_features[i], motion_features[i])
            ts = audio_features[i].timestamp
            scores.append((ts, s))

        # Extract above-threshold regions
        moments: list[Moment] = []
        in_region = False
        region_start = 0.0
        region_scores: list[float] = []

        for ts, s in scores:
            if s >= self.threshold:
                if not in_region:
                    region_start = ts
                    region_scores = []
                    in_region = True
                region_scores.append(s)
            else:
                if in_region:
                    moments.append(
                        self._finalise_moment(region_start, ts, region_scores)
                    )
                    in_region = False

        if in_region and region_scores:
            last_ts = scores[-1][0]
            moments.append(
                self._finalise_moment(region_start, last_ts + 1.0, region_scores)
            )

        # Merge moments that are too close
        merged = self._merge_close(moments, min_gap_seconds)
        return merged

    def compute_motion_features(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray | None = None,
        timestamp: float = 0.0,
    ) -> MotionFeatures:
        """Compute motion features from consecutive frames.

        Uses simple frame-differencing as an optical-flow proxy.
        """
        if prev_frame is None or frame.shape != prev_frame.shape:
            return MotionFeatures(timestamp=timestamp)

        diff = np.abs(frame.astype(float) - prev_frame.astype(float))
        intensity = float(np.mean(diff) / 255.0)
        # High-motion region clustering as density proxy
        threshold_mask = diff.mean(axis=-1) > 30 if diff.ndim == 3 else diff > 30
        density = float(np.mean(threshold_mask))

        return MotionFeatures(
            timestamp=timestamp,
            motion_intensity=min(1.0, intensity * 3.0),
            player_density=min(1.0, density * 2.0),
            celebration_probability=0.0,  # would need pose detection
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _finalise_moment(
        start: float,
        end: float,
        scores: list[float],
    ) -> Moment:
        peak = max(scores) if scores else 0.0
        return Moment(
            timestamp=start,
            duration=max(1.0, end - start),
            excitement_score=round(peak, 3),
            audio_score=round(peak, 3),
            motion_score=0.0,
            confidence=round(sum(scores) / len(scores), 3) if scores else 0.0,
        )

    @staticmethod
    def _merge_close(moments: list[Moment], min_gap: float) -> list[Moment]:
        if not moments:
            return []
        merged: list[Moment] = [moments[0]]
        for m in moments[1:]:
            prev = merged[-1]
            prev_end = prev.timestamp + prev.duration
            if m.timestamp - prev_end < min_gap:
                # Extend previous moment
                new_end = m.timestamp + m.duration
                merged[-1] = Moment(
                    timestamp=prev.timestamp,
                    duration=new_end - prev.timestamp,
                    excitement_score=max(prev.excitement_score, m.excitement_score),
                    audio_score=max(prev.audio_score, m.audio_score),
                    motion_score=max(prev.motion_score, m.motion_score),
                    confidence=max(prev.confidence, m.confidence),
                )
            else:
                merged.append(m)
        return merged
