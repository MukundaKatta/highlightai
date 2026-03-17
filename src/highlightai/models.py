"""Pydantic data models for HighlightAI."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Sport(str, Enum):
    """Supported sports."""

    SOCCER = "soccer"
    FOOTBALL = "football"
    BASKETBALL = "basketball"
    CRICKET = "cricket"
    HOCKEY = "hockey"
    BASEBALL = "baseball"
    TENNIS = "tennis"


class EventType(str, Enum):
    """Types of notable events across sports."""

    GOAL = "goal"
    TOUCHDOWN = "touchdown"
    DUNK = "dunk"
    WICKET = "wicket"
    HOME_RUN = "home_run"
    THREE_POINTER = "three_pointer"
    PENALTY = "penalty"
    FREE_KICK = "free_kick"
    SAVE = "save"
    FOUL = "foul"
    CELEBRATION = "celebration"
    FIGHT = "fight"
    ACE = "ace"
    SLAM = "slam"
    BREAKAWAY = "breakaway"
    SACK = "sack"
    INTERCEPTION = "interception"
    CATCH = "catch"


class AudioFeatures(BaseModel):
    """Audio features extracted from a time window."""

    timestamp: float = Field(..., description="Centre timestamp in seconds")
    duration: float = Field(default=1.0, description="Window duration in seconds")
    crowd_noise_level: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Normalised crowd noise intensity"
    )
    commentary_excitement: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Commentary excitement score"
    )
    is_crowd_roar: bool = Field(default=False, description="Crowd roar detected")
    is_commentary_peak: bool = Field(
        default=False, description="Commentary peak detected"
    )


class MotionFeatures(BaseModel):
    """Visual motion features for a frame or window."""

    timestamp: float = 0.0
    motion_intensity: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall motion magnitude"
    )
    player_density: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Player clustering score"
    )
    celebration_probability: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Celebration detection score"
    )


class GameFootage(BaseModel):
    """Metadata describing a game recording."""

    path: str = Field(default="", description="File path or URI")
    sport: Sport = Sport.SOCCER
    duration_seconds: float = Field(default=0.0, ge=0.0)
    fps: float = Field(default=30.0, gt=0.0)
    resolution: tuple[int, int] = Field(default=(1920, 1080))
    home_team: str = ""
    away_team: str = ""


class Moment(BaseModel):
    """A detected moment of interest in the footage."""

    timestamp: float = Field(..., description="Start time in seconds")
    duration: float = Field(default=5.0, description="Duration in seconds")
    event_type: Optional[EventType] = None
    excitement_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Composite excitement score"
    )
    audio_score: float = Field(default=0.0, ge=0.0, le=1.0)
    motion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    description: str = ""


class Highlight(BaseModel):
    """A clip extracted from footage for the highlight reel."""

    moment: Moment
    clip_start: float = Field(..., description="Clip start time with context")
    clip_end: float = Field(..., description="Clip end time with context")
    rank: int = Field(default=0, description="Position in the reel")
    transition_in: str = Field(default="cut", description="Transition type entering")
    transition_out: str = Field(default="cut", description="Transition type exiting")


class HighlightReel(BaseModel):
    """A compiled highlight reel."""

    footage: GameFootage
    highlights: list[Highlight] = Field(default_factory=list)
    total_duration: float = 0.0
    num_moments: int = 0
    average_excitement: float = 0.0

    def compute_stats(self) -> None:
        """Recompute aggregate statistics."""
        self.num_moments = len(self.highlights)
        self.total_duration = sum(
            h.clip_end - h.clip_start for h in self.highlights
        )
        if self.highlights:
            self.average_excitement = sum(
                h.moment.excitement_score for h in self.highlights
            ) / len(self.highlights)
