"""AudioAnalyzer - detect crowd roar and commentary excitement peaks."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import signal

from highlightai.models import AudioFeatures


class AudioAnalyzer:
    """Analyse audio tracks to detect crowd noise and commentary peaks.

    In production this would operate on raw PCM / spectrograms; here it works
    on pre-extracted amplitude envelopes or simulated signals.
    """

    def __init__(
        self,
        crowd_threshold: float = 0.7,
        commentary_threshold: float = 0.75,
        sample_rate: float = 44100.0,
        window_seconds: float = 1.0,
    ) -> None:
        self.crowd_threshold = crowd_threshold
        self.commentary_threshold = commentary_threshold
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyse_signal(
        self,
        audio_signal: np.ndarray,
        sample_rate: Optional[float] = None,
    ) -> list[AudioFeatures]:
        """Analyse an audio waveform and return per-window features.

        Parameters
        ----------
        audio_signal:
            1-D array of audio samples (mono, normalised to [-1, 1]).
        sample_rate:
            Samples per second; defaults to self.sample_rate.
        """
        sr = sample_rate or self.sample_rate
        window_samples = int(sr * self.window_seconds)
        num_windows = max(1, len(audio_signal) // window_samples)
        features: list[AudioFeatures] = []

        for i in range(num_windows):
            start = i * window_samples
            end = min(start + window_samples, len(audio_signal))
            chunk = audio_signal[start:end]

            crowd_noise = self._estimate_crowd_noise(chunk, sr)
            commentary = self._estimate_commentary_excitement(chunk, sr)

            timestamp = (start + end) / 2 / sr
            features.append(
                AudioFeatures(
                    timestamp=timestamp,
                    duration=self.window_seconds,
                    crowd_noise_level=crowd_noise,
                    commentary_excitement=commentary,
                    is_crowd_roar=crowd_noise >= self.crowd_threshold,
                    is_commentary_peak=commentary >= self.commentary_threshold,
                )
            )

        return features

    def detect_peaks(
        self,
        features: list[AudioFeatures],
        min_gap_seconds: float = 5.0,
    ) -> list[AudioFeatures]:
        """Return only the peak excitement moments, enforcing a minimum gap."""
        if not features:
            return []

        scores = np.array(
            [
                max(f.crowd_noise_level, f.commentary_excitement)
                for f in features
            ]
        )
        # Use scipy to find peaks
        peak_indices, _ = signal.find_peaks(
            scores,
            height=self.crowd_threshold,
            distance=max(1, int(min_gap_seconds / self.window_seconds)),
        )
        return [features[i] for i in peak_indices]

    # ------------------------------------------------------------------
    # Feature extractors (simplified)
    # ------------------------------------------------------------------

    def _estimate_crowd_noise(
        self,
        chunk: np.ndarray,
        sr: float,
    ) -> float:
        """Estimate crowd noise level from an audio chunk.

        Uses RMS energy in the 200-4000 Hz band as a proxy.
        """
        if len(chunk) < 4:
            return 0.0
        # Band-pass via FFT
        fft = np.fft.rfft(chunk)
        freqs = np.fft.rfftfreq(len(chunk), d=1.0 / sr)
        mask = (freqs >= 200) & (freqs <= 4000)
        band_energy = np.sqrt(np.mean(np.abs(fft[mask]) ** 2)) if mask.any() else 0.0
        # Normalise to [0, 1] with a soft ceiling
        normalised = float(np.tanh(band_energy * 3.0))
        return min(1.0, max(0.0, normalised))

    def _estimate_commentary_excitement(
        self,
        chunk: np.ndarray,
        sr: float,
    ) -> float:
        """Estimate commentary excitement from an audio chunk.

        Uses energy in the speech band (300-3000 Hz) combined with variance
        (excited speech has higher amplitude variance).
        """
        if len(chunk) < 4:
            return 0.0
        fft = np.fft.rfft(chunk)
        freqs = np.fft.rfftfreq(len(chunk), d=1.0 / sr)
        mask = (freqs >= 300) & (freqs <= 3000)
        band_energy = np.sqrt(np.mean(np.abs(fft[mask]) ** 2)) if mask.any() else 0.0
        variance = float(np.var(chunk))
        score = float(np.tanh(band_energy * 2.0 + variance * 5.0))
        return min(1.0, max(0.0, score))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def combined_score(self, features: AudioFeatures) -> float:
        """Return a single combined audio excitement score."""
        return min(
            1.0,
            0.6 * features.crowd_noise_level + 0.4 * features.commentary_excitement,
        )
