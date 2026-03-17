"""Tests for MomentRanker."""

import pytest

from highlightai.editor.ranker import MomentRanker
from highlightai.models import EventType, Moment, Sport


class TestMomentRanker:
    def setup_method(self):
        self.ranker = MomentRanker(Sport.SOCCER)

    def test_goal_ranks_higher_than_foul(self):
        goal = Moment(
            timestamp=100.0,
            event_type=EventType.GOAL,
            excitement_score=0.8,
            confidence=0.9,
        )
        foul = Moment(
            timestamp=200.0,
            event_type=EventType.FOUL,
            excitement_score=0.8,
            confidence=0.9,
        )
        assert self.ranker.rank_score(goal) > self.ranker.rank_score(foul)

    def test_rank_moments_ordering(self):
        moments = [
            Moment(timestamp=100.0, event_type=EventType.FOUL, excitement_score=0.5, confidence=0.5),
            Moment(timestamp=200.0, event_type=EventType.GOAL, excitement_score=0.9, confidence=0.9),
            Moment(timestamp=300.0, event_type=EventType.SAVE, excitement_score=0.7, confidence=0.7),
        ]
        ranked = self.ranker.rank_moments(moments)
        assert ranked[0].event_type == EventType.GOAL

    def test_top_n(self):
        moments = [
            Moment(timestamp=float(i), excitement_score=i / 10.0, confidence=0.5)
            for i in range(10)
        ]
        top = self.ranker.top_n(moments, 3)
        assert len(top) == 3
        # Highest excitement should be first
        assert top[0].excitement_score >= top[1].excitement_score

    def test_annotate_scores(self):
        moments = [
            Moment(timestamp=50.0, event_type=EventType.GOAL, excitement_score=0.85, confidence=0.8),
        ]
        annotated = self.ranker.annotate_scores(moments)
        assert len(annotated) == 1
        moment, score = annotated[0]
        assert 0 <= score <= 1.0

    def test_unknown_event_type_gets_default(self):
        # An event type not in the soccer importance map
        m = Moment(
            timestamp=10.0,
            event_type=EventType.DUNK,  # not a soccer event
            excitement_score=0.7,
            confidence=0.8,
        )
        score = self.ranker.rank_score(m)
        assert 0 <= score <= 1.0
