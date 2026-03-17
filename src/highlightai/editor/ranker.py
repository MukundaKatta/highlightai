"""MomentRanker - score and order highlight moments."""

from __future__ import annotations

from highlightai.models import EventType, Moment, Sport


# Event importance weights by sport
_EVENT_IMPORTANCE: dict[Sport, dict[EventType, float]] = {
    Sport.SOCCER: {
        EventType.GOAL: 1.0,
        EventType.PENALTY: 0.9,
        EventType.SAVE: 0.7,
        EventType.FREE_KICK: 0.5,
        EventType.CELEBRATION: 0.6,
        EventType.FOUL: 0.3,
    },
    Sport.FOOTBALL: {
        EventType.TOUCHDOWN: 1.0,
        EventType.INTERCEPTION: 0.85,
        EventType.SACK: 0.7,
        EventType.CATCH: 0.6,
        EventType.CELEBRATION: 0.5,
    },
    Sport.BASKETBALL: {
        EventType.DUNK: 1.0,
        EventType.THREE_POINTER: 0.85,
        EventType.BREAKAWAY: 0.7,
        EventType.FOUL: 0.4,
        EventType.CELEBRATION: 0.5,
    },
    Sport.CRICKET: {
        EventType.WICKET: 1.0,
        EventType.CATCH: 0.8,
        EventType.CELEBRATION: 0.5,
    },
    Sport.HOCKEY: {
        EventType.GOAL: 1.0,
        EventType.SAVE: 0.7,
        EventType.FIGHT: 0.8,
        EventType.BREAKAWAY: 0.65,
    },
    Sport.BASEBALL: {
        EventType.HOME_RUN: 1.0,
        EventType.CATCH: 0.7,
        EventType.CELEBRATION: 0.5,
    },
    Sport.TENNIS: {
        EventType.ACE: 0.7,
        EventType.SLAM: 0.8,
        EventType.CELEBRATION: 1.0,
    },
}


class MomentRanker:
    """Score and rank moments by excitement and sport-specific importance.

    The final rank score combines:
    - Raw excitement score (audio + motion + celebration)
    - Event type importance weight
    - Detection confidence
    """

    def __init__(
        self,
        sport: Sport = Sport.SOCCER,
        excitement_weight: float = 0.5,
        importance_weight: float = 0.3,
        confidence_weight: float = 0.2,
    ) -> None:
        self.sport = sport
        self.excitement_weight = excitement_weight
        self.importance_weight = importance_weight
        self.confidence_weight = confidence_weight
        self._importance = _EVENT_IMPORTANCE.get(sport, {})

    def rank_score(self, moment: Moment) -> float:
        """Compute a composite ranking score for a moment."""
        importance = 0.5  # default for unknown event types
        if moment.event_type and moment.event_type in self._importance:
            importance = self._importance[moment.event_type]

        score = (
            self.excitement_weight * moment.excitement_score
            + self.importance_weight * importance
            + self.confidence_weight * moment.confidence
        )
        return round(min(1.0, score), 4)

    def rank_moments(self, moments: list[Moment]) -> list[Moment]:
        """Sort moments by rank score descending.

        Returns a new list; original is not mutated.
        """
        scored = [(self.rank_score(m), m) for m in moments]
        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored]

    def top_n(self, moments: list[Moment], n: int) -> list[Moment]:
        """Return the top N moments."""
        ranked = self.rank_moments(moments)
        return ranked[:n]

    def annotate_scores(
        self,
        moments: list[Moment],
    ) -> list[tuple[Moment, float]]:
        """Return moments paired with their rank scores."""
        return [(m, self.rank_score(m)) for m in moments]
