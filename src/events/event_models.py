"""
Event Models — Pydantic models for structured gameplay events.

Defines the data schema for the event timeline JSON output.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field
import json
from pathlib import Path


class EventType(str, Enum):
    """Types of detectable badminton events."""
    RALLY_START = "rally_start"
    SMASH = "smash"
    DROP_SHOT = "drop_shot"
    LONG_RALLY = "long_rally"
    POINT_WON = "point_won"
    OUT_OF_BOUNDS = "out_of_bounds"
    NET_FAULT = "net_fault"


class Intensity(str, Enum):
    """Intensity levels for events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


class Event(BaseModel):
    """A single gameplay event with timestamp and metadata."""
    model_config = ConfigDict(use_enum_values=True)

    timestamp: float = Field(..., description="Time in seconds when the event occurred")
    event: EventType = Field(..., description="Type of event detected")
    by: Optional[str] = Field(None, description="Player who triggered the event")
    server: Optional[str] = Field(None, description="Serving player (for rally_start)")
    intensity: Intensity = Field(Intensity.MEDIUM, description="Event intensity level")
    rally_length: Optional[int] = Field(None, description="Number of shots in the rally")
    rally_duration: Optional[float] = Field(None, description="Duration of rally in seconds")
    velocity: Optional[float] = Field(None, description="Shuttle velocity at event time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional event data")


class MatchState(BaseModel):
    """Current match scoring state."""
    player_a_name: str = "Player A"
    player_b_name: str = "Player B"
    player_a_score: int = 0
    player_b_score: int = 0
    current_set: int = 1
    serving: Optional[str] = None


class EventTimeline(BaseModel):
    """Complete event timeline for a video."""
    video_path: str = ""
    video_duration: float = 0.0
    player_a: str = "Player A"
    player_b: str = "Player B"
    total_events: int = 0
    total_points: int = 0
    events: List[Event] = Field(default_factory=list)
    match_state: MatchState = Field(default_factory=MatchState)

    def add_event(self, event: Event):
        """Add an event to the timeline."""
        self.events.append(event)
        self.total_events = len(self.events)
        if event.event == EventType.POINT_WON:
            self.total_points += 1

    def to_json(self, indent: int = 2) -> str:
        """Serialize timeline to JSON string."""
        return self.model_dump_json(indent=indent)

    def save(self, output_path: str):
        """Save timeline to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "EventTimeline":
        """Load timeline from a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)

    def summary(self) -> str:
        """Return a human-readable summary of the timeline."""
        lines = [
            f"{'='*50}",
            f"  EVENT TIMELINE SUMMARY",
            f"{'='*50}",
            f"  Video: {self.video_path}",
            f"  Duration: {self.video_duration:.1f}s",
            f"  Players: {self.player_a} vs {self.player_b}",
            f"  Total Events: {self.total_events}",
            f"  Total Points: {self.total_points}",
            f"{'─'*50}",
        ]

        # Event type breakdown
        from collections import Counter
        event_counts = Counter(e.event for e in self.events)
        lines.append("  Event Breakdown:")
        for etype, count in event_counts.most_common():
            lines.append(f"    {etype}: {count}")

        # Score
        lines.append(f"{'─'*50}")
        lines.append(
            f"  Score: {self.match_state.player_a_name} {self.match_state.player_a_score} - "
            f"{self.match_state.player_b_score} {self.match_state.player_b_name}"
        )
        lines.append(f"{'='*50}")

        return "\n".join(lines)
