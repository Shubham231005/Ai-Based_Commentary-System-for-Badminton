"""
Debug Event Logger — Frame-by-frame CSV log of all detection and rule evaluations.

Produces a machine-readable CSV file for post-analysis in Excel, Pandas, etc.
Each row = one frame, with all detection state, rule evaluations, and scoring.
"""

import csv
from pathlib import Path
from typing import List, Optional, Any
import logging

logger = logging.getLogger(__name__)

CSV_HEADERS = [
    "frame",
    "timestamp",
    "phase",
    "velocity",
    "decay_rate",
    "vy",
    "shuttle_x",
    "shuttle_y",
    "is_predicted",
    "player_a_x",
    "player_a_y",
    "player_b_x",
    "player_b_y",
    "rally_active",
    "hit_count",
    "last_hitter",
    "events_fired",
    "score_a",
    "score_b",
    "landing_signals",
    "attribution",
]


class DebugEventLogger:
    """
    Logs per-frame debug data to a CSV file.

    Each row captures the complete detection + tracking + rule evaluation
    state for one frame, making it easy to find exactly when and why
    rules trigger.
    """

    def __init__(self, output_path: str = "output/debug_log.csv"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_HEADERS)
        self._row_count = 0
        logger.info(f"Debug CSV logger initialized: {self.output_path}")

    def log_frame(
        self,
        frame_index: int,
        features: Any,
        events: List[Any],
        score_a: int = 0,
        score_b: int = 0,
        landing_signals: Optional[List] = None,
        last_attribution: Optional[dict] = None,
    ):
        """Log a single frame's state to the CSV."""
        # Shuttle position
        sx, sy = "", ""
        if features.shuttle_position:
            sx = f"{features.shuttle_position[0]:.1f}"
            sy = f"{features.shuttle_position[1]:.1f}"

        # Player positions
        pax, pay = "", ""
        if features.player_a_position:
            pax = f"{features.player_a_position[0]:.1f}"
            pay = f"{features.player_a_position[1]:.1f}"
        pbx, pby = "", ""
        if features.player_b_position:
            pbx = f"{features.player_b_position[0]:.1f}"
            pby = f"{features.player_b_position[1]:.1f}"

        # Events
        event_names = []
        for evt in events:
            name = evt.event if isinstance(evt.event, str) else evt.event.value
            event_names.append(name)
        events_str = "|".join(event_names) if event_names else ""

        # Landing signals
        landing_str = ""
        if landing_signals:
            landing_str = "|".join(f"{n}:{w*c:.2f}" for n, w, c in landing_signals)

        # Attribution
        attr_str = ""
        if last_attribution:
            attr_str = (
                f"{last_attribution.get('winner', '?')}|"
                f"{last_attribution.get('reason', '?')}|"
                f"{last_attribution.get('confidence', 0):.2f}"
            )

        row = [
            frame_index,
            f"{features.timestamp:.3f}",
            features.shuttle_phase,
            f"{features.shuttle_velocity:.2f}",
            f"{features.velocity_decay_rate:.3f}",
            f"{features.shuttle_vy:.2f}",
            sx, sy,
            "1" if features.is_predicted else "0",
            pax, pay,
            pbx, pby,
            "1" if features.rally_active else "0",
            features.rally_hit_count,
            features.last_hitter or "",
            events_str,
            score_a,
            score_b,
            landing_str,
            attr_str,
        ]

        self._writer.writerow(row)
        self._row_count += 1

    def finalize(self):
        """Close the CSV file."""
        if self._file:
            self._file.close()
            logger.info(f"Debug CSV saved: {self.output_path} ({self._row_count} rows)")
