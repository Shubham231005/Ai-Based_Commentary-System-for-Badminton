"""
Commentary Models — Pydantic data models for commentary output.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CommentaryLine(BaseModel):
    """A single commentary line tied to a match event."""
    timestamp: float = Field(..., description="Event timestamp in seconds")
    text: str = Field(..., description="Commentary text (Hinglish)")
    event_type: str = Field("", description="Source event type")
    intensity: str = Field("medium", description="Delivery intensity")
    score_context: str = Field("", description="Score at this point, e.g. '3-2'")


class CommentaryTimeline(BaseModel):
    """Complete commentary output for a match."""
    video_path: str = ""
    player_a: str = "Player A"
    player_b: str = "Player B"
    persona: str = "hinglish_excited"
    total_lines: int = 0
    lines: List[CommentaryLine] = Field(default_factory=list)
    match_summary: str = ""

    def add_line(self, line: CommentaryLine):
        self.lines.append(line)
        self.total_lines = len(self.lines)

    def to_text(self) -> str:
        """Human-readable commentary script."""
        parts = [
            f"{'='*60}",
            f"  COMMENTARY SCRIPT — {self.player_a} vs {self.player_b}",
            f"  Persona: {self.persona}",
            f"{'='*60}",
            "",
        ]
        for line in self.lines:
            mins = int(line.timestamp // 60)
            secs = line.timestamp % 60
            ts = f"[{mins:02d}:{secs:04.1f}]"
            parts.append(f"{ts}  {line.text}")

        if self.match_summary:
            parts.extend(["", f"{'─'*60}", "MATCH SUMMARY:", self.match_summary])

        parts.append(f"\n{'='*60}")
        return "\n".join(parts)

    def save(self, output_dir: str = "output"):
        """Save both JSON and human-readable text."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = out / "commentary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

        # Text
        txt_path = out / "commentary.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(self.to_text())

        return str(json_path), str(txt_path)

    @classmethod
    def load(cls, path: str) -> "CommentaryTimeline":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))
