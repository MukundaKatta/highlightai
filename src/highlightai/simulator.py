"""Simulator - generate synthetic game data for testing and demos."""

from __future__ import annotations

from typing import Optional

import numpy as np

from highlightai.models import (
    AudioFeatures,
    EventType,
    GameFootage,
    Moment,
    MotionFeatures,
    Sport,
)


# Typical event times as fraction of game duration, per sport
_TYPICAL_EVENT_FRACTIONS: dict[Sport, list[tuple[float, EventType, str]]] = {
    Sport.SOCCER: [
        (0.12, EventType.FREE_KICK, "Free kick from 25 yards"),
        (0.25, EventType.GOAL, "Opening goal - header from corner"),
        (0.33, EventType.FOUL, "Yellow card tackle"),
        (0.45, EventType.SAVE, "Brilliant diving save"),
        (0.55, EventType.GOAL, "Counter-attack goal"),
        (0.68, EventType.PENALTY, "Penalty awarded"),
        (0.70, EventType.GOAL, "Penalty converted"),
        (0.82, EventType.SAVE, "Goal-line clearance"),
        (0.92, EventType.GOAL, "Last-minute equaliser"),
    ],
    Sport.FOOTBALL: [
        (0.10, EventType.CATCH, "Deep pass completed"),
        (0.22, EventType.TOUCHDOWN, "Rushing touchdown"),
        (0.35, EventType.SACK, "Blindside sack"),
        (0.48, EventType.INTERCEPTION, "Pick-six interception"),
        (0.60, EventType.CATCH, "One-handed catch"),
        (0.75, EventType.TOUCHDOWN, "Hail Mary touchdown"),
        (0.90, EventType.CELEBRATION, "Victory formation"),
    ],
    Sport.BASKETBALL: [
        (0.08, EventType.DUNK, "Fast-break alley-oop"),
        (0.20, EventType.THREE_POINTER, "Deep three from logo"),
        (0.35, EventType.BREAKAWAY, "Coast-to-coast layup"),
        (0.50, EventType.DUNK, "Poster dunk"),
        (0.65, EventType.THREE_POINTER, "And-one three-pointer"),
        (0.80, EventType.DUNK, "Windmill dunk"),
        (0.95, EventType.CELEBRATION, "Buzzer-beater celebration"),
    ],
    Sport.CRICKET: [
        (0.15, EventType.WICKET, "Bowled through the gate"),
        (0.30, EventType.CATCH, "Diving slip catch"),
        (0.50, EventType.WICKET, "LBW decision"),
        (0.70, EventType.CATCH, "Outfield boundary catch"),
        (0.90, EventType.WICKET, "Stumping to win"),
    ],
}


class GameSimulator:
    """Generate synthetic audio and motion features for a simulated game."""

    def __init__(
        self,
        sport: Sport = Sport.SOCCER,
        duration_minutes: float = 90.0,
        home_team: str = "Home",
        away_team: str = "Away",
        seed: Optional[int] = None,
    ) -> None:
        self.sport = sport
        self.duration_seconds = duration_minutes * 60.0
        self.home_team = home_team
        self.away_team = away_team
        self.rng = np.random.default_rng(seed)

    def footage_metadata(self) -> GameFootage:
        return GameFootage(
            path="simulated_game.mp4",
            sport=self.sport,
            duration_seconds=self.duration_seconds,
            fps=30.0,
            home_team=self.home_team,
            away_team=self.away_team,
        )

    def generate_audio_features(
        self,
        window_seconds: float = 1.0,
    ) -> list[AudioFeatures]:
        """Generate synthetic audio features with spikes at event times."""
        events = _TYPICAL_EVENT_FRACTIONS.get(self.sport, [])
        event_times = [
            frac * self.duration_seconds for frac, _, _ in events
        ]

        num_windows = int(self.duration_seconds / window_seconds)
        features: list[AudioFeatures] = []

        for i in range(num_windows):
            ts = (i + 0.5) * window_seconds
            # Baseline noise
            crowd = float(self.rng.uniform(0.05, 0.25))
            commentary = float(self.rng.uniform(0.05, 0.20))

            # Spike near events
            for et in event_times:
                dist = abs(ts - et)
                if dist < 8.0:
                    spike = max(0.0, 1.0 - dist / 8.0)
                    crowd = min(1.0, crowd + spike * self.rng.uniform(0.5, 0.9))
                    commentary = min(
                        1.0, commentary + spike * self.rng.uniform(0.4, 0.8)
                    )

            features.append(
                AudioFeatures(
                    timestamp=ts,
                    duration=window_seconds,
                    crowd_noise_level=round(crowd, 3),
                    commentary_excitement=round(commentary, 3),
                    is_crowd_roar=crowd >= 0.7,
                    is_commentary_peak=commentary >= 0.75,
                )
            )

        return features

    def generate_motion_features(
        self,
        window_seconds: float = 1.0,
    ) -> list[MotionFeatures]:
        """Generate synthetic motion features with spikes at event times."""
        events = _TYPICAL_EVENT_FRACTIONS.get(self.sport, [])
        event_times = [
            frac * self.duration_seconds for frac, _, _ in events
        ]

        num_windows = int(self.duration_seconds / window_seconds)
        features: list[MotionFeatures] = []

        for i in range(num_windows):
            ts = (i + 0.5) * window_seconds
            intensity = float(self.rng.uniform(0.1, 0.35))
            density = float(self.rng.uniform(0.1, 0.3))
            celebration = 0.0

            for j, et in enumerate(event_times):
                dist = abs(ts - et)
                if dist < 5.0:
                    spike = max(0.0, 1.0 - dist / 5.0)
                    intensity = min(1.0, intensity + spike * self.rng.uniform(0.4, 0.8))
                    density = min(1.0, density + spike * self.rng.uniform(0.2, 0.5))
                    # Celebration comes slightly after the event
                    if 0 < (ts - et) < 6.0:
                        celebration = min(
                            1.0, spike * self.rng.uniform(0.3, 0.8)
                        )

            features.append(
                MotionFeatures(
                    timestamp=ts,
                    motion_intensity=round(intensity, 3),
                    player_density=round(density, 3),
                    celebration_probability=round(celebration, 3),
                )
            )

        return features

    def generate_expected_moments(self) -> list[Moment]:
        """Return the ground-truth moments for the simulated game."""
        events = _TYPICAL_EVENT_FRACTIONS.get(self.sport, [])
        moments: list[Moment] = []
        for frac, event_type, desc in events:
            ts = frac * self.duration_seconds
            moments.append(
                Moment(
                    timestamp=ts,
                    duration=5.0,
                    event_type=event_type,
                    excitement_score=0.9,
                    audio_score=0.85,
                    motion_score=0.8,
                    confidence=0.9,
                    description=desc,
                )
            )
        return moments
