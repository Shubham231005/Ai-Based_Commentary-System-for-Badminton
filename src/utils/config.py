"""
Configuration loader for Antigravity system
Handles YAML config and environment variables
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

class Config:
    """Central configuration management"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Default to config/config.yaml
            base_dir = Path(__file__).parent.parent.parent
            config_path = base_dir / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._load_config()
        self._load_env_vars()
    
    def _load_config(self):
        """Load YAML configuration file"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def _load_env_vars(self):
        """Load API keys from environment variables"""
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.google_tts_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        
        # Validate required keys
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
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
        """Get configuration for a specific sport"""
        return self.config.get('sports', {}).get(sport.lower(), {})
    
    def get_language_config(self, language: str) -> Dict:
        """Get configuration for a specific language"""
        return self.config.get('languages', {}).get(language.lower(), {})
    
    def get_persona_config(self, persona: str) -> Dict:
        """Get configuration for a specific persona"""
        return self.config.get('personas', {}).get(persona.lower(), {})

# Global config instance
_config = None

def get_config() -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config
