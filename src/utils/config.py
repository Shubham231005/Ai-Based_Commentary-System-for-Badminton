"""
Configuration loader for Antigravity system.
Handles YAML config and environment variables.
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Load environment variables
load_dotenv()


class Config:
    """Central configuration management."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            base_dir = Path(__file__).parent.parent.parent
            config_path = base_dir / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self._load_config()
        self._load_env_vars()

    def _load_config(self):
        """Load YAML configuration file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def _load_env_vars(self):
        """Load API keys from environment variables (optional in Phase 1)."""
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.google_tts_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

        # Phase 1 does NOT require API keys (pure CV pipeline)
        # API keys only needed in Phase 2 (LLM) and Phase 3 (TTS)
        if not self.gemini_api_key:
            import logging
            logging.getLogger(__name__).info(
                "GEMINI_API_KEY not set — Phase 1 (CV pipeline) works without it. "
                "Set it when you need Phase 2 (LLM commentary)."
            )

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        Example: config.get('video.fps') returns config['video']['fps']
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_sport_config(self, sport: str) -> Dict:
        """Get configuration for a specific sport."""
        return self.config.get('sport', {}) if sport.lower() == 'badminton' else {}

    def get_event_rules(self) -> Dict:
        """Get event detection rules."""
        return self.config.get('event_rules', {})

    def get_intensity_config(self) -> Dict:
        """Get intensity scoring configuration."""
        return self.config.get('intensity', {})

    def get_court_config(self) -> Dict:
        """Get court geometry configuration."""
        return self.config.get('court', {})


# Global config instance
_config: Optional[Config] = None


def get_config(config_path: str = None) -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config
