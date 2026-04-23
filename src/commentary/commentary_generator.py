"""
Commentary Generator — Hinglish Sports Commentary Engine.

Generates authentic Indian-style Hinglish commentary (Hindi + English mix)
from the event timeline. Uses Gemini API when available, falls back to
rich template-based commentary when API key is not set.

Style: Star Sports Hindi commentary vibe — energetic, natural Hindi-English
mix, cricket-commentary feel applied to badminton.
"""

import json
import random
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ..events.event_models import EventTimeline, Event
from .commentary_models import CommentaryLine, CommentaryTimeline

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
#  HINGLISH COMMENTARY TEMPLATES (Fallback when no API key)
# ════════════════════════════════════════════════════════════════

RALLY_START_TEMPLATES = [
    "ओये होये! {server} ने serve उछाली — ये rally नहीं, ये तो Mumbai local है, रुकने का नाम ही नहीं!",
    "चलिए जी! {server} की serve — जैसे Sachin ने toss जीत लिया हो, game ON!",
    "बेटा, {server} serve कर रहे हैं — पूरा stadium साँस रोके बैठा है!",
    "और शुरू होता है नया chapter! {server} ने serve फेंकी जैसे बॉलीवुड में entry scene हो!",
    "{server} की serve — tension ऐसी जैसे India-Pakistan final का last over!",
    "Action! {server} ready हैं — popcorn निकालो भाई, show शुरू हुआ!",
    "ताली बजाओ! {server} ने serve की — ये rally तो blockbuster होने वाली है!",
    "देखो देखो! {server} ने shuttle उछाली — अब होगा असली drama!",
]

SMASH_TEMPLATES = {
    "low": [
        "{player} ने smash मारा — ठीक-ठाक था, जैसे chai में चीनी कम पड़ गई!",
        "अरे {player} का smash — controlled है, जैसे Dhoni last over में खेलता है!",
        "{player} का smash — speed कम but placement ऐसा जैसे GPS लगा हो!",
    ],
    "medium": [
        "SMASH! {player} ने दबा के मारा! {velocity:.0f} की speed — बेटा ये तो rocket था!",
        "क्या shot है {player} का! Smash ऐसा जैसे Virat ने cover drive मारी हो!",
        "वाह जी वाह! {player} ने full power smash — {velocity:.0f} speed, opponent बेचारा देखता रह गया!",
        "धांय! {player} का smash — ये shuttle नहीं, ये तो गोला बरूद है भाई!",
        "{player} ने ऐसा smash मारा — opponent सोच रहा है 'ये हुआ क्या अभी?'",
    ],
    "high": [
        "ओ माँ! {player} ON FIRE है! {velocity:.0f} speed — ये smash नहीं, ये तो बिजली गिरी!",
        "अरे बापरे! {player} ने तोड़ दिया! {velocity:.0f} की speed — ये तो missile था भाई!",
        "धमाका! {player} का smash — जैसे Diwali में atom bomb फोड़ दिया हो!",
        "OH MY GOD! {player} से devastating smash! Opponent के पास chance ही नहीं था!",
        "BOOM! {player} का thunderous smash! जैसे शेर ने दहाड़ लगाई — crowd पागल!",
    ],
    "maximum": [
        "ये SMASH नहीं, ये तो TSUNAMI है! {player} ने आग लगा दी!! {velocity:.0f} speed!! 🔥🔥",
        "ओये होये! {player} ने ऐसा मारा — commentary box तक आ गई shuttle! {velocity:.0f} speed!!",
        "{player} ने KYA MAARA HAI! {velocity:.0f} speed — ये shuttle नहीं, ये तो Brahmos missile थी!!",
        "SMAAASH!! {player} ने तोड़ डाला! Stadium हिल गया! {velocity:.0f} speed — ये है REAL POWER!!",
        "भाई ये match नहीं, ये तो DESTRUCTION है! {player} का {velocity:.0f} speed smash — UNBELIEVABLE!!",
    ],
}

DROP_SHOT_TEMPLATES = {
    "low": [
        "{player} ने चुपके से drop shot डाल दिया — जैसे बिल्ली दूध पी गई!",
        "ओहो! {player} की चालबाज़ी! Drop shot — बिल्कुल net के पास!",
    ],
    "medium": [
        "क्या finesse है! {player} का drop shot — ऐसा touch जैसे माँ ने सर पे हाथ फेरा हो!",
        "अरे वाह {player}! Drop shot with perfect touch — opponent को छक्का दे दिया!",
        "DECEPTION! {player} ने drop मारा जब सबको smash की उम्मीद थी — बेटा ये badminton नहीं, ये शतरंज है shuttle के साथ!",
    ],
    "high": [
        "GENIUS! {player} का drop shot — shuttle net के ऊपर से चूम के गई! क्या touch है!",
        "Masterclass! {player} ने ऐसा drop मारा — opponent reach भी नहीं कर पाया! World class!",
        "वाह {player}! Drop shot ऐसा मारा कि shuttle net से लिपट गई — BEAUTIFUL!",
    ],
    "maximum": [
        "PERFECTION! {player} का drop shot — ये तो art है भाई! Shuttle ने net को kiss किया!",
        "IMPOSSIBLE drop shot! {player} — ये shot सिर्फ legends मारते हैं! गेंद उधर, दिमाग इधर — game over, picture superhit!",
    ],
}

LONG_RALLY_TEMPLATES = [
    "ओये होये! ये rally नहीं, ये तो Mumbai local है — रुकने का नाम ही नहीं ले रही! {hits} shots in {duration:.1f} seconds!",
    "WHAT A RALLY! {hits} shots! {duration:.1f} seconds! दोनों ऐसे खेल रहे हैं जैसे दो शेर एक ही जंगल में शिकार कर रहे हों!",
    "ये rally ख़तम होने का नाम ही नहीं ले रही! {hits} shots — {duration:.1f} seconds — ये तो Mahabharata चल रही है!",
    "EPIC RALLY! {hits} exchanges! {duration:.1f} seconds — बेटा ये badminton नहीं, ये तो stamina का World Cup है!",
    "अरे भाई! {hits} shots already! {duration:.1f} seconds! दोनों players ऐसे भिड़ रहे हैं जैसे Baahubali vs Bhallaaldeva!",
]

POINT_WON_TEMPLATES = {
    "low": [
        "और point गया {player} को! Score: {score} — चलो आगे बढ़ो!",
        "{player} ले गए ये point! {score} — calm and composed!",
        "Easy point for {player}! Score: {score} — जैसे गरम चाय का पहला sip!",
    ],
    "medium": [
        "POINT! {player} ने ले लिया! {score}! अच्छी rally थी — crowd ने enjoy किया!",
        "{player} takes the point! {score}! बोहोत अच्छा खेला दोनों ने!",
        "{player} के खाते में एक और point! {score}! Game interesting हो रही है!",
        "Well played! {player} wins this one! {score}! मज़ा आ रहा है!",
    ],
    "high": [
        "BRILLIANT POINT! {player} ने कमाल कर दिया! {score}! {rally_len} shots की rally — crowd पागल!",
        "WHAT A POINT! {player}! {score}! {rally_len} shots के बाद finally decide हुआ!",
        "{player} WINS IT after {rally_len} incredible shots! {score}! ये तो Bollywood climax था!",
        "क्या GAME है! {player} takes it! {score}! {duration:.1f} seconds की rally — goosebumps!",
    ],
    "maximum": [
        "SENSATIONAL POINT! {player} ने जीत लिया! {score}! {rally_len} shots का EPIC rally — ये तो history है!",
        "UN-BE-LIEVABLE! {player}! {score}! {rally_len} shots और {duration:.1f} seconds — picture SUPERHIT!",
        "MATCH OF THE CENTURY VIBES! {player} takes it! {score}! {rally_len} shots! Stadium में सन्नाटा फिर तालियाँ!",
    ],
}

MATCH_SUMMARY_TEMPLATES = [
    "क्या match था ये! {pa} vs {pb}, final score {score}! "
    "Total {total_points} points — हर rally में Bollywood जैसा drama! "
    "{winner} ने शानदार performance दी! {duration:.0f} seconds का धमाकेदार match!",

    "और ये match ख़तम होता है! {pa} vs {pb}: {score}! "
    "{total_points} points, हर point पे दिल धड़का! "
    "{winner} आज का KING/QUEEN है — मज़ा आ गया देखके! 🏸🔥",
]

# ═══════════════════════════════════════════════════════════════
#  GEMINI SYSTEM PROMPT (for when API key is available)
# ═══════════════════════════════════════════════════════════════

HINGLISH_SYSTEM_PROMPT = """You are Navjot Singh Sidhu reborn as a badminton commentator — LIVE on Star Sports Hindi.

YOUR STYLE:
- Hinglish: Hindi in DEVANAGARI script mixed freely with English words
- Think Sidhu's one-liners + Aakash Chopra energy + Bollywood flair
- ALWAYS use Devanagari for Hindi ("ओये होये!" not "Oye hoye!")
- English words stay in English ("smash", "rally", "shot", "speed")
- Use cricket analogies, Bollywood references, desi metaphors
- Energy level should match intensity: low = witty one-liners, high/maximum = FULL PAGAL mode
- Exclamations: "ओये होये!", "अरे बापरे!", "क्या बात है!", "वाह जी वाह!"
- Crowd references: "Stadium हिल गया!", "Crowd पागल हो गई!"
- Keep it SHORT and PUNCHY — max 2 sentences per event
- For smashes, mention speed and use weapon/missile analogies
- For long rallies, compare to Mumbai local / Mahabharata / Bollywood fights
- For points won, give score update and celebrate

EXAMPLE LINES (match this vibe):
- "ओये होये! ये rally नहीं, ये तो Mumbai local है — रुकने का नाम ही नहीं ले रही!"
- "बेटा ये badminton नहीं, ये शतरंज है shuttle के साथ!"
- "गेंद उधर, दिमाग इधर — game over, picture superhit!"
- "ऐसे खेल रहे हैं जैसे दो शेर एक ही जंगल में शिकार कर रहे हों!"

IMPORTANT RULES:
- Don't question scoring or detection — just commentate
- Short rallies = quick clinical points; long rallies = GO CRAZY
- Make every line quotable and memorable

OUTPUT FORMAT:
Return a JSON array of objects with:
- "timestamp": event timestamp (number)
- "text": your commentary line (string, Devanagari Hindi + English mix)
- "event_type": the event type (string)

Example:
[
  {"timestamp": 1.5, "text": "चलिए जी! Player A ने serve उछाली — पूरा stadium साँस रोके बैठा है!", "event_type": "rally_start"},
  {"timestamp": 3.2, "text": "ओ माँ! Player B का SMASH — 45 speed — ये shuttle नहीं, ये तो Brahmos missile थी!", "event_type": "smash"}
]

Only return the JSON array, nothing else."""


class CommentaryGenerator:
    """
    Generates Hinglish commentary from an EventTimeline.

    Uses Gemini API when available, falls back to rich
    template-based commentary otherwise.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        commentary_cfg = config.get("commentary", {})

        self.persona = commentary_cfg.get("persona", "hinglish_excited")
        self.model_name = commentary_cfg.get("model", "gemini-2.0-flash")
        self.temperature = commentary_cfg.get("temperature", 0.85)
        self.include_summary = commentary_cfg.get("include_match_summary", True)

        # Try to load Gemini
        self._gemini_model = None
        self._init_gemini()

    def _init_gemini(self):
        """Try to initialize Gemini API client."""
        try:
            import google.generativeai as genai
            import os
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.info(
                    "GEMINI_API_KEY not found — using template-based Hinglish commentary. "
                    "Set the key in .env for LLM-powered commentary."
                )
                return

            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    top_p=0.95,
                    max_output_tokens=8192,
                ),
                system_instruction=HINGLISH_SYSTEM_PROMPT,
            )
            logger.info(f"Gemini initialized: {self.model_name} (Hinglish commentary mode)")

        except ImportError:
            logger.info(
                "google-generativeai not installed — using template-based commentary. "
                "Install with: pip install google-generativeai"
            )
        except Exception as e:
            logger.warning(f"Gemini init failed: {e} — falling back to templates")

    def generate(self, timeline: EventTimeline) -> CommentaryTimeline:
        """
        Generate commentary for the entire event timeline.

        Filters to key moments only — badminton is fast, so we only
        commentate on points, big smashes, long rallies, and key starts.
        Target: ~70-80 lines for a full match.
        """
        logger.info(
            f"Generating Hinglish commentary for {len(timeline.events)} events "
            f"({timeline.player_a} vs {timeline.player_b})"
        )

        # Filter to key moments only
        key_events = self._filter_key_events(timeline)
        logger.info(f"Filtered to {len(key_events)} key moments (from {len(timeline.events)} total)")

        # Create a filtered timeline for generation
        filtered_timeline = EventTimeline(
            video_path=timeline.video_path,
            video_duration=timeline.video_duration,
            player_a=timeline.player_a,
            player_b=timeline.player_b,
            match_state=timeline.match_state,
        )
        for evt in key_events:
            filtered_timeline.add_event(evt)

        commentary = CommentaryTimeline(
            video_path=timeline.video_path,
            player_a=timeline.player_a,
            player_b=timeline.player_b,
            persona=self.persona,
        )

        if self._gemini_model:
            lines = self._generate_gemini(filtered_timeline)
        else:
            lines = self._generate_templates(filtered_timeline)

        for line in lines:
            commentary.add_line(line)

        # Match summary
        if self.include_summary:
            summary = self._generate_summary(timeline)
            commentary.match_summary = summary

        logger.info(f"Commentary generated: {commentary.total_lines} lines")
        return commentary

    def _filter_key_events(self, timeline: EventTimeline) -> list:
        """
        Filter events to only the important moments worth commentating.

        Badminton is FAST — events every 0.3s. Real commentary only
        covers key moments. Target: ~70-80 lines for a full match.

        Rules:
        - ALWAYS: point_won (every score update matters)
        - ALWAYS: long_rally (hype moments)
        - SELECTIVE: smash — only maximum intensity OR velocity > 90
        - SELECTIVE: drop_shot — only maximum intensity (rare brilliance)
        - SELECTIVE: rally_start — first one + every 8th rally
        - Minimum gap: 3s between non-point commentary lines
        """
        key_events = []
        rally_count = 0
        last_commentary_ts = -10.0
        MIN_GAP = 3.0  # seconds between commentary (keeps it natural)

        for evt in timeline.events:
            event_type = evt.event if isinstance(evt.event, str) else evt.event.value
            intensity = evt.intensity if isinstance(evt.intensity, str) else evt.intensity.value

            include = False
            points_seen = getattr(self, '_points_seen', 0)

            if event_type == "point_won":
                self._points_seen = getattr(self, '_points_seen', 0) + 1
                # Interesting points: high intensity, every 3rd, or after a gap
                if intensity in ("high", "maximum"):
                    include = True
                elif self._points_seen % 3 == 0:
                    include = True  # Every 3rd point for score updates
                elif evt.timestamp - last_commentary_ts > 5.0:
                    include = True  # Been quiet, update score

            elif event_type == "long_rally":
                # Always — these build excitement
                include = True

            elif event_type == "smash":
                # Only monster smashes
                if intensity == "maximum":
                    include = True
                elif evt.velocity and evt.velocity > 90:
                    include = True

            elif event_type == "drop_shot":
                # Only brilliant drop shots
                if intensity == "maximum":
                    include = True

            elif event_type == "rally_start":
                rally_count += 1
                # First rally + periodically
                if rally_count == 1:
                    include = True
                elif rally_count % 8 == 0:
                    include = True
                elif evt.timestamp - last_commentary_ts > 15.0:
                    include = True  # Been quiet, mention new rally

            # Enforce minimum gap (point_won always goes through)
            if include and event_type != "point_won":
                if evt.timestamp - last_commentary_ts < MIN_GAP:
                    include = False

            if include:
                key_events.append(evt)
                last_commentary_ts = evt.timestamp

        return key_events

    # ── Gemini-powered generation (single call) ────────────────

    def _generate_gemini(self, timeline: EventTimeline) -> List[CommentaryLine]:
        """Generate commentary using ONE Gemini API call."""
        # Build the user prompt with full event timeline
        prompt = self._build_gemini_prompt(timeline)

        try:
            logger.info("Sending event timeline to Gemini (single API call)...")
            response = self._gemini_model.generate_content(prompt)
            raw_text = response.text.strip()

            # Parse JSON from response (strip markdown code blocks if present)
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[1]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3].strip()

            parsed = json.loads(raw_text)
            lines = []
            for item in parsed:
                lines.append(CommentaryLine(
                    timestamp=float(item["timestamp"]),
                    text=item["text"],
                    event_type=item.get("event_type", ""),
                    intensity=item.get("intensity", "medium"),
                ))

            logger.info(f"Gemini returned {len(lines)} commentary lines")
            return lines

        except Exception as e:
            logger.warning(f"Gemini API failed: {e} — falling back to templates")
            return self._generate_templates(timeline)

    def _build_gemini_prompt(self, timeline: EventTimeline) -> str:
        """Build the user prompt for Gemini with full event timeline."""
        # Score tracking context
        score_a, score_b = 0, 0

        events_for_prompt = []
        for evt in timeline.events:
            entry = {
                "timestamp": evt.timestamp,
                "event": evt.event if isinstance(evt.event, str) else evt.event.value,
                "by": evt.by,
                "server": evt.server,
                "intensity": evt.intensity if isinstance(evt.intensity, str) else evt.intensity.value,
            }
            if evt.velocity:
                entry["velocity"] = round(evt.velocity, 1)
            if evt.rally_length:
                entry["rally_length"] = evt.rally_length
            if evt.rally_duration:
                entry["rally_duration"] = round(evt.rally_duration, 1)

            # Track score for context
            if (evt.event if isinstance(evt.event, str) else evt.event.value) == "point_won":
                if evt.by == timeline.player_a:
                    score_a += 1
                else:
                    score_b += 1
                entry["score"] = f"{score_a}-{score_b}"

            events_for_prompt.append(entry)

        prompt = f"""Here is the COMPLETE event timeline for a badminton match:

MATCH: {timeline.player_a} vs {timeline.player_b}
DURATION: {timeline.video_duration:.0f} seconds
TOTAL EVENTS: {len(timeline.events)}
TOTAL POINTS: {timeline.total_points}

EVENTS:
{json.dumps(events_for_prompt, indent=2)}

Generate Hinglish commentary for EVERY event. Remember:
- Match the intensity level of each event
- Keep score updates on point_won events
- Build excitement during long rallies
- Short, punchy, natural Hinglish — like you're really commentating LIVE
- Return ONLY a JSON array as specified in the system prompt"""

        return prompt

    # ── Template-based generation (fallback) ───────────────────

    def _generate_templates(self, timeline: EventTimeline) -> List[CommentaryLine]:
        """Generate commentary using Hinglish templates (no API needed)."""
        lines = []
        score_a, score_b = 0, 0

        for evt in timeline.events:
            event_type = evt.event if isinstance(evt.event, str) else evt.event.value
            intensity = evt.intensity if isinstance(evt.intensity, str) else evt.intensity.value
            player = evt.by or evt.server or "Player"

            text = ""

            if event_type == "rally_start":
                server = evt.server or "Player"
                text = random.choice(RALLY_START_TEMPLATES).format(server=server)

            elif event_type == "smash":
                templates = SMASH_TEMPLATES.get(intensity, SMASH_TEMPLATES["medium"])
                vel = evt.velocity or 0
                text = random.choice(templates).format(player=player, velocity=vel)

            elif event_type == "drop_shot":
                templates = DROP_SHOT_TEMPLATES.get(intensity, DROP_SHOT_TEMPLATES["medium"])
                text = random.choice(templates).format(player=player)

            elif event_type == "long_rally":
                hits = evt.rally_length or 0
                dur = evt.rally_duration or 0
                text = random.choice(LONG_RALLY_TEMPLATES).format(
                    hits=hits, duration=dur
                )

            elif event_type == "point_won":
                if evt.by == timeline.player_a:
                    score_a += 1
                else:
                    score_b += 1
                score_str = f"{score_a}-{score_b}"
                rally_len = evt.rally_length or 0
                dur = evt.rally_duration or 0

                templates = POINT_WON_TEMPLATES.get(intensity, POINT_WON_TEMPLATES["medium"])
                text = random.choice(templates).format(
                    player=player, score=score_str,
                    rally_len=rally_len, duration=dur,
                )

            if text:
                lines.append(CommentaryLine(
                    timestamp=evt.timestamp,
                    text=text,
                    event_type=event_type,
                    intensity=intensity,
                    score_context=f"{score_a}-{score_b}",
                ))

        return lines

    # ── Match summary ──────────────────────────────────────────

    def _generate_summary(self, timeline: EventTimeline) -> str:
        """Generate end-of-match summary."""
        ms = timeline.match_state
        score_str = f"{ms.player_a_score}-{ms.player_b_score}"
        winner = ms.player_a_name if ms.player_a_score > ms.player_b_score else ms.player_b_name

        if self._gemini_model:
            try:
                prompt = (
                    f"Write a 2-3 sentence Hinglish match summary for:\n"
                    f"{ms.player_a_name} vs {ms.player_b_name}, Score: {score_str}\n"
                    f"Total points: {timeline.total_points}, Duration: {timeline.video_duration:.0f}s\n"
                    f"Winner: {winner}\n"
                    f"Keep it natural Hinglish, celebratory, mention key stats. Plain text only."
                )
                response = self._gemini_model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                logger.warning(f"Gemini summary failed: {e}")

        # Template fallback
        return random.choice(MATCH_SUMMARY_TEMPLATES).format(
            pa=ms.player_a_name,
            pb=ms.player_b_name,
            score=score_str,
            winner=winner,
            total_points=timeline.total_points,
            duration=timeline.video_duration,
        )


# ═══════════════════════════════════════════════════════════════
#  Standalone CLI entrypoint
# ═══════════════════════════════════════════════════════════════

def main():
    """Run commentary generation on an existing event timeline."""
    import argparse
    import sys
    import yaml

    # Fix Windows console encoding for emoji/Unicode
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Load .env file for API keys
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not installed, rely on system env vars

    parser = argparse.ArgumentParser(description="Generate Hinglish commentary from event timeline")
    parser.add_argument("--timeline", required=True, help="Path to event_timeline.json")
    parser.add_argument("--persona", default="hinglish_excited", help="Commentary persona")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    # Load config
    config_path = args.config or str(Path(__file__).parent.parent.parent / "config" / "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config.setdefault("commentary", {})["persona"] = args.persona

    # Load timeline
    timeline = EventTimeline.load(args.timeline)
    logger.info(f"Loaded timeline: {timeline.total_events} events, {timeline.total_points} points")

    # Generate
    generator = CommentaryGenerator(config)
    commentary = generator.generate(timeline)

    # Save
    json_path, txt_path = commentary.save(args.output)
    print(f"\nCommentary saved:")
    print(f"   JSON: {json_path}")
    print(f"   Text: {txt_path}")
    print(f"\nPreview (first 10 lines):")
    for line in commentary.lines[:10]:
        mins = int(line.timestamp // 60)
        secs = line.timestamp % 60
        print(f"  [{mins:02d}:{secs:04.1f}] {line.text}")
    if len(commentary.lines) > 10:
        print(f"  ... and {len(commentary.lines) - 10} more lines")


if __name__ == "__main__":
    main()

