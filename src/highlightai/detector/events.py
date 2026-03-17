"""EventDetector - identify key sport-specific moments."""

from __future__ import annotations

from typing import Optional

import numpy as np

from highlightai.models import AudioFeatures, EventType, Moment, MotionFeatures, Sport


# Event signatures: sport -> list of (event_type, required_audio_min, motion_min, description)
_EVENT_SIGNATURES: dict[Sport, list[tuple[EventType, float, float, str]]] = {
    Sport.SOCCER: [
        (EventType.GOAL, 0.85, 0.5, "Goal scored"),
        (EventType.PENALTY, 0.7, 0.3, "Penalty kick"),
        (EventType.FREE_KICK, 0.5, 0.4, "Free kick"),
        (EventType.SAVE, 0.6, 0.6, "Goalkeeper save"),
        (EventType.CELEBRATION, 0.8, 0.3, "Goal celebration"),
        (EventType.FOUL, 0.4, 0.5, "Foul committed"),
    ],
    Sport.FOOTBALL: [
        (EventType.TOUCHDOWN, 0.85, 0.5, "Touchdown"),
        (EventType.INTERCEPTION, 0.7, 0.6, "Interception"),
        (EventType.SACK, 0.6, 0.7, "Quarterback sack"),
        (EventType.CATCH, 0.5, 0.5, "Big catch"),
        (EventType.CELEBRATION, 0.75, 0.3, "Touchdown celebration"),
    ],
    Sport.BASKETBALL: [
        (EventType.DUNK, 0.8, 0.7, "Slam dunk"),
        (EventType.THREE_POINTER, 0.75, 0.3, "Three-pointer"),
        (EventType.BREAKAWAY, 0.6, 0.8, "Fast break"),
        (EventType.FOUL, 0.4, 0.5, "Flagrant foul"),
        (EventType.CELEBRATION, 0.7, 0.3, "Celebration"),
    ],
    Sport.CRICKET: [
        (EventType.WICKET, 0.8, 0.4, "Wicket taken"),
        (EventType.CATCH, 0.7, 0.5, "Spectacular catch"),
        (EventType.CELEBRATION, 0.75, 0.3, "Wicket celebration"),
    ],
    Sport.HOCKEY: [
        (EventType.GOAL, 0.85, 0.5, "Goal scored"),
        (EventType.SAVE, 0.6, 0.6, "Big save"),
        (EventType.FIGHT, 0.7, 0.8, "Fight on ice"),
        (EventType.BREAKAWAY, 0.6, 0.7, "Breakaway"),
    ],
    Sport.BASEBALL: [
        (EventType.HOME_RUN, 0.85, 0.4, "Home run"),
        (EventType.CATCH, 0.65, 0.6, "Diving catch"),
        (EventType.CELEBRATION, 0.7, 0.3, "Home run celebration"),
    ],
    Sport.TENNIS: [
        (EventType.ACE, 0.6, 0.3, "Ace served"),
        (EventType.SLAM, 0.7, 0.5, "Powerful slam"),
        (EventType.CELEBRATION, 0.75, 0.3, "Match point celebration"),
    ],
}


class EventDetector:
    """Detect sport-specific events from audio and motion features.

    Each sport has a set of event signatures defining the minimum audio and
    motion thresholds.  The detector scans time-aligned features and emits
    Moment objects when signatures are matched.
    """

    def __init__(self, sport: Sport = Sport.SOCCER) -> None:
        self.sport = sport
        self.signatures = _EVENT_SIGNATURES.get(sport, [])

    def detect_events(
        self,
        audio_features: list[AudioFeatures],
        motion_features: list[MotionFeatures] | None = None,
        min_gap_seconds: float = 10.0,
    ) -> list[Moment]:
        """Scan features and return detected events."""
        if motion_features is None:
            motion_features = [
                MotionFeatures(timestamp=a.timestamp) for a in audio_features
            ]

        n = min(len(audio_features), len(motion_features))
        raw_events: list[Moment] = []

        for i in range(n):
            af = audio_features[i]
            mf = motion_features[i]
            audio_level = max(af.crowd_noise_level, af.commentary_excitement)
            motion_level = mf.motion_intensity

            for event_type, audio_min, motion_min, desc in self.signatures:
                if audio_level >= audio_min and motion_level >= motion_min:
                    confidence = min(
                        1.0,
                        (audio_level / audio_min + motion_level / motion_min) / 2.0,
                    )
                    raw_events.append(
                        Moment(
                            timestamp=af.timestamp,
                            duration=5.0,
                            event_type=event_type,
                            excitement_score=round(
                                0.5 * audio_level + 0.5 * motion_level, 3
                            ),
                            audio_score=round(audio_level, 3),
                            motion_score=round(motion_level, 3),
                            confidence=round(confidence, 3),
                            description=desc,
                        )
                    )
                    break  # one event per time window (highest priority first)

        # Deduplicate / enforce minimum gap
        return self._deduplicate(raw_events, min_gap_seconds)

    def supported_events(self) -> list[EventType]:
        """List event types for the current sport."""
        return [sig[0] for sig in self.signatures]

    @staticmethod
    def all_sports() -> list[Sport]:
        return list(Sport)

    @staticmethod
    def events_for_sport(sport: Sport) -> list[tuple[EventType, str]]:
        sigs = _EVENT_SIGNATURES.get(sport, [])
        return [(s[0], s[3]) for s in sigs]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(
        events: list[Moment],
        min_gap: float,
    ) -> list[Moment]:
        if not events:
            return []
        # Sort by timestamp
        events.sort(key=lambda m: m.timestamp)
        result: list[Moment] = [events[0]]
        for e in events[1:]:
            prev = result[-1]
            if e.timestamp - prev.timestamp >= min_gap:
                result.append(e)
            elif e.excitement_score > prev.excitement_score:
                result[-1] = e  # replace with higher-scoring event
        return result
