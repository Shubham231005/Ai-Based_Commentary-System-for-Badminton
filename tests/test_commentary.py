"""
Tests for Phase 2: Hinglish Commentary Engine.

Tests template-based commentary generation, rally grouping,
model serialization, and prompt construction.
"""

import pytest
import json
from src.events.event_models import Event, EventTimeline, MatchState, EventType, Intensity
from src.commentary.commentary_models import CommentaryLine, CommentaryTimeline
from src.commentary.commentary_generator import CommentaryGenerator


# ── Fixtures ───────────────────────────────────────────────────

def _make_timeline() -> EventTimeline:
    """Create a realistic mini event timeline for testing."""
    timeline = EventTimeline(
        video_path="test_video.mp4",
        video_duration=30.0,
        player_a="Lakshya",
        player_b="Srikanth",
    )

    events = [
        Event(timestamp=1.5, event="rally_start", server="Lakshya", intensity="low"),
        Event(timestamp=2.0, event="smash", by="Srikanth", intensity="medium", velocity=45.2, rally_length=1),
        Event(timestamp=3.0, event="drop_shot", by="Lakshya", intensity="low", velocity=4.0),
        Event(timestamp=4.5, event="point_won", by="Srikanth", intensity="medium", rally_length=3, rally_duration=3.0),

        Event(timestamp=6.0, event="rally_start", server="Srikanth", intensity="low"),
        Event(timestamp=7.0, event="smash", by="Srikanth", intensity="high", velocity=92.0, rally_length=4),
        Event(timestamp=8.0, event="long_rally", intensity="high", rally_length=10, rally_duration=5.0),
        Event(timestamp=9.5, event="smash", by="Lakshya", intensity="maximum", velocity=110.0, rally_length=15),
        Event(timestamp=10.0, event="point_won", by="Lakshya", intensity="high", rally_length=15, rally_duration=8.0),

        Event(timestamp=12.0, event="rally_start", server="Lakshya", intensity="low"),
        Event(timestamp=13.0, event="drop_shot", by="Lakshya", intensity="medium", velocity=3.5),
        Event(timestamp=14.0, event="point_won", by="Lakshya", intensity="low", rally_length=2, rally_duration=2.0),
    ]

    for evt in events:
        timeline.add_event(evt)

    timeline.match_state = MatchState(
        player_a_name="Lakshya",
        player_b_name="Srikanth",
        player_a_score=2,
        player_b_score=1,
    )

    return timeline


def _make_config() -> dict:
    return {
        "commentary": {
            "enabled": True,
            "persona": "hinglish_excited",
            "model": "gemini-2.0-flash",
            "temperature": 0.85,
            "include_match_summary": True,
        }
    }


# ── Template Generation Tests ─────────────────────────────────

class TestTemplateCommentary:
    """Test template-based commentary (no API key)."""

    def test_generates_lines_for_all_events(self):
        """Should produce one commentary line per event."""
        config = _make_config()
        gen = CommentaryGenerator(config)
        timeline = _make_timeline()

        commentary = gen.generate(timeline)

        assert commentary.total_lines == len(timeline.events)
        assert len(commentary.lines) == len(timeline.events)

    def test_timestamps_match_events(self):
        """Commentary timestamps should match source events."""
        gen = CommentaryGenerator(_make_config())
        commentary = gen.generate(_make_timeline())

        for line, evt in zip(commentary.lines, _make_timeline().events):
            assert line.timestamp == evt.timestamp

    def test_commentary_text_not_empty(self):
        """Every line should have non-empty text."""
        gen = CommentaryGenerator(_make_config())
        commentary = gen.generate(_make_timeline())

        for line in commentary.lines:
            assert len(line.text) > 0
            assert line.text.strip() != ""

    def test_event_types_preserved(self):
        """Commentary lines should have correct event types."""
        gen = CommentaryGenerator(_make_config())
        commentary = gen.generate(_make_timeline())
        timeline = _make_timeline()

        for line, evt in zip(commentary.lines, timeline.events):
            expected = evt.event if isinstance(evt.event, str) else evt.event.value
            assert line.event_type == expected

    def test_score_tracking(self):
        """Score context should update correctly through events."""
        gen = CommentaryGenerator(_make_config())
        commentary = gen.generate(_make_timeline())

        # Find point_won events
        point_lines = [l for l in commentary.lines if l.event_type == "point_won"]
        assert len(point_lines) == 3

        # Score should progress: 0-1, 1-1, 2-1
        assert point_lines[0].score_context == "0-1"
        assert point_lines[1].score_context == "1-1"
        assert point_lines[2].score_context == "2-1"

    def test_intensity_preserved(self):
        """Commentary lines should preserve intensity from events."""
        gen = CommentaryGenerator(_make_config())
        commentary = gen.generate(_make_timeline())

        for line, evt in zip(commentary.lines, _make_timeline().events):
            expected = evt.intensity if isinstance(evt.intensity, str) else evt.intensity.value
            assert line.intensity == expected

    def test_match_summary_generated(self):
        """Should include a match summary when enabled."""
        gen = CommentaryGenerator(_make_config())
        commentary = gen.generate(_make_timeline())

        assert len(commentary.match_summary) > 0

    def test_no_summary_when_disabled(self):
        """Should skip summary when disabled in config."""
        config = _make_config()
        config["commentary"]["include_match_summary"] = False
        gen = CommentaryGenerator(config)
        commentary = gen.generate(_make_timeline())

        assert commentary.match_summary == ""

    def test_player_names_in_commentary(self):
        """Player names should appear in commentary text."""
        gen = CommentaryGenerator(_make_config())
        commentary = gen.generate(_make_timeline())

        all_text = " ".join(l.text for l in commentary.lines)
        assert "Lakshya" in all_text or "Srikanth" in all_text

    def test_smash_velocity_in_text(self):
        """Smash commentary should reference the velocity."""
        gen = CommentaryGenerator(_make_config())
        commentary = gen.generate(_make_timeline())

        smash_lines = [l for l in commentary.lines if l.event_type == "smash"]
        assert len(smash_lines) >= 1
        # At least one smash should mention a velocity number
        has_velocity = any("45" in l.text or "92" in l.text or "110" in l.text for l in smash_lines)
        assert has_velocity


# ── Model Serialization Tests ─────────────────────────────────

class TestCommentaryModels:
    """Test CommentaryTimeline serialization."""

    def test_to_text_output(self):
        """Should produce human-readable text."""
        gen = CommentaryGenerator(_make_config())
        commentary = gen.generate(_make_timeline())

        text = commentary.to_text()
        assert "COMMENTARY SCRIPT" in text
        assert "Lakshya" in text
        assert "Srikanth" in text

    def test_json_round_trip(self, tmp_path):
        """JSON save/load should preserve all data."""
        gen = CommentaryGenerator(_make_config())
        commentary = gen.generate(_make_timeline())

        # Save
        json_path = tmp_path / "test_commentary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(commentary.model_dump_json(indent=2))

        # Load
        loaded = CommentaryTimeline.load(str(json_path))

        assert loaded.total_lines == commentary.total_lines
        assert loaded.player_a == "Lakshya"
        assert loaded.player_b == "Srikanth"
        assert len(loaded.lines) == len(commentary.lines)

    def test_empty_timeline(self):
        """Should handle an empty event timeline gracefully."""
        timeline = EventTimeline(
            video_path="empty.mp4",
            video_duration=0.0,
            player_a="A",
            player_b="B",
        )

        gen = CommentaryGenerator(_make_config())
        commentary = gen.generate(timeline)

        assert commentary.total_lines == 0
        assert len(commentary.lines) == 0


# ── Prompt Construction Tests ─────────────────────────────────

class TestPromptConstruction:
    """Test Gemini prompt building (without actual API call)."""

    def test_prompt_contains_all_events(self):
        """Built prompt should include all event data."""
        gen = CommentaryGenerator(_make_config())
        timeline = _make_timeline()
        prompt = gen._build_gemini_prompt(timeline)

        assert "Lakshya" in prompt
        assert "Srikanth" in prompt
        assert "rally_start" in prompt
        assert "smash" in prompt
        assert "point_won" in prompt
        assert "drop_shot" in prompt
        assert "long_rally" in prompt

    def test_prompt_contains_score_context(self):
        """Prompt should include score updates on point_won events."""
        gen = CommentaryGenerator(_make_config())
        timeline = _make_timeline()
        prompt = gen._build_gemini_prompt(timeline)

        # Should have score strings in the prompt
        assert "0-1" in prompt  # after first point
        assert "1-1" in prompt  # after second point

    def test_prompt_contains_velocity(self):
        """Prompt should include velocity for smash events."""
        gen = CommentaryGenerator(_make_config())
        timeline = _make_timeline()
        prompt = gen._build_gemini_prompt(timeline)

        assert "45.2" in prompt
        assert "92.0" in prompt
