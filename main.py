"""
Antigravity — Event-Driven Sports Commentary Engine

CLI entry point for Phase 1: Vision-based Event Timeline Generation.

Usage:
    python main.py --video <path_to_video> --player-a "Name" --player-b "Name"
    python main.py --video datasets/sample_match.mp4 --player-a "Lin Dan" --player-b "Lee Chong Wei"
    python main.py --video datasets/sample_match.mp4 --debug  # Save annotated debug frames
"""

import argparse
import logging
import sys
from pathlib import Path

from src.utils.config import Config
from src.pipeline import CommentaryPipeline


def setup_logging(verbose: bool = False):
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Suppress noisy loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(
        description="Antigravity — Event-Driven Sports Commentary Engine (Phase 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --video datasets/sample_match.mp4
  python main.py --video match.mp4 --player-a "Lin Dan" --player-b "Lee Chong Wei" --fps 5
  python main.py --video match.mp4 --debug --verbose
        """,
    )

    parser.add_argument(
        "--video", "-v",
        required=True,
        help="Path to the badminton match video file",
    )
    parser.add_argument(
        "--player-a", "-a",
        default="Player A",
        help="Name of Player A (near court / bottom of frame)",
    )
    parser.add_argument(
        "--player-b", "-b",
        default="Player B",
        help="Name of Player B (far court / top of frame)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for the event timeline JSON (default: output/event_timeline.json)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Override target FPS for frame extraction (default: from config)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Max video duration to process in seconds",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save annotated debug frames to output/debug_frames/",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--debug-live",
        action="store_true",
        help="Show live debug visualization window (quit with 'q')",
    )
    parser.add_argument(
        "--commentary",
        action="store_true",
        default=None,
        help="Enable Hinglish commentary generation (default: uses config)",
    )
    parser.add_argument(
        "--no-commentary",
        action="store_true",
        help="Skip commentary generation",
    )
    parser.add_argument(
        "--persona",
        default=None,
        help="Commentary persona: hinglish_excited, analytical, casual",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to custom config.yaml (default: config/config.yaml)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger("main")

    # Validate video path
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    # Load config
    try:
        config = Config(config_path=args.config)
    except ValueError as e:
        # Allow running without API keys in Phase 1
        logger.warning(f"Config warning (non-fatal for Phase 1): {e}")
        import yaml
        config_path = args.config or Path(__file__).parent / "config" / "config.yaml"
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = type('Config', (), {'config': config_dict, 'get': lambda self, k, d=None: self.config.get(k, d)})()

    # Override config with CLI args
    cfg = getattr(config, 'config', None) or {}
    if args.fps:
        cfg.setdefault("video", {})["fps"] = args.fps
    if args.max_duration:
        cfg.setdefault("video", {})["max_duration"] = args.max_duration
    if args.debug:
        cfg.setdefault("output", {})["debug_frames"] = True

    # Banner
    print(r"""
     _    _   _ _____ ___ ____ ____      ___     _____ _______   __
    / \  | \ | |_   _|_ _/ ___|  _ \    / \ \   / /_ _|_   _\ \ / /
   / _ \ |  \| | | |  | | |  _| |_) |  / _ \ \ / / | |  | |  \ V /
  / ___ \| |\  | | |  | | |_| |  _ <  / ___ \ V /  | |  | |   | |
 /_/   \_\_| \_| |_| |___\____|_| \_\/_/   \_\_/  |___| |_|   |_|

 Event-Driven Sports Commentary Engine — Phase 1
    """)

    video_cfg = cfg.get("video", {}) or {}
    logger.info(f"Video:    {video_path}")
    logger.info(f"Players:  {args.player_a} vs {args.player_b}")
    logger.info(f"FPS:      {video_cfg.get('fps', 10)}")

    # Create and run pipeline
    debug_mode = args.debug or args.debug_live

    # Commentary: CLI flags override config
    commentary_enabled = True  # default
    if args.no_commentary:
        commentary_enabled = False
    elif args.commentary:
        commentary_enabled = True

    # Override persona if specified
    if args.persona:
        cfg.setdefault("commentary", {})["persona"] = args.persona

    pipeline = CommentaryPipeline(
        config=cfg,
        player_a=args.player_a,
        player_b=args.player_b,
        debug_mode=debug_mode,
        debug_live=args.debug_live,
        commentary_enabled=commentary_enabled,
    )

    if debug_mode:
        logger.info("Debug visualization mode ENABLED")
        logger.info("  → output/debug_output.mp4 (annotated video)")
        logger.info("  → output/debug_log.csv (per-frame data)")
        if args.debug_live:
            logger.info("  → Live window active (press 'q' to close)")

    output_path = args.output or "output/event_timeline.json"
    timeline = pipeline.run(
        video_path=str(video_path),
        output_path=output_path,
    )

    # Final output
    print(f"\n✅ Event timeline saved to: {output_path}")
    print(f"📊 {timeline.total_events} events detected, {timeline.total_points} points scored")

    return 0


if __name__ == "__main__":
    sys.exit(main())
