import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Set
from dotenv import load_dotenv

# Load environment variables from .env file in development
load_dotenv()

class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass

class Config:
    def __init__(self):
        # Determine base directory for relative file paths
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Load configuration from YAML file
        try:
            with open(os.path.join(self.BASE_DIR, "config.yaml"), "r") as f:
                self._config = yaml.safe_load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")
        
        # Extract configuration sections
        self.MODEL_CONFIG = self._config.get("models", {})
        self.MODEL_MAPPING = self._config.get("model_mapping", {})
        self.THRESHOLDS = self._config.get("thresholds", {})
        self.MEDICAL_TERMS = self._config.get("medical_terms", {})
        
        # Environment variable overrides
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", self.MODEL_CONFIG.get("default"))
        
        # API Keys
        _raw_api_keys = os.getenv("X_API_KEYS", "")
        self.VALID_API_KEYS: Set[str] = {k.strip() for k in _raw_api_keys.split(",") if k.strip()}
        
        # Frontend URL for CORS
        self.FRONTEND_URL = os.getenv("FRONTEND_URL")
        if not self.FRONTEND_URL and os.getenv("ENVIRONMENT") != "development":
            print("⚠️ Warning: FRONTEND_URL not configured, using development defaults")
            
        # In development, allow all origins, otherwise restrict to specified frontend
        if os.getenv("ENVIRONMENT") == "development":
            self.ALLOWED_ORIGINS = ["*"]
        else:
            self.ALLOWED_ORIGINS = [self.FRONTEND_URL] if self.FRONTEND_URL else ["http://localhost:8080"]
        
        # Logging configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        
        # Print configuration summary on initialization
        print(f"🔧 Configuration loaded from {os.path.join(self.BASE_DIR, 'config.yaml')}")
        print(f"🧠 Using embedding model: {self.EMBEDDING_MODEL}")
        print(f"🔑 API Keys configured: {len(self.VALID_API_KEYS)}")
        print(f"🌐 CORS allowed origins: {self.ALLOWED_ORIGINS}")
        print(f"📝 Logging level: {self.LOG_LEVEL}")
    
    def get_medical_abbreviations(self) -> Dict[str, str]:
        """Get dictionary of medical abbreviations and their expansions."""
        return self.MEDICAL_TERMS.get("abbreviations", {})
    
    def get_medical_synonyms(self) -> Dict[str, List[str]]:
        """Get dictionary of medical terms and their synonyms."""
        return self.MEDICAL_TERMS.get("synonyms", {})
    
    def get_threshold_for_module(self, module: str) -> float:
        """Get the validation threshold for a specific module."""
        if module not in self.THRESHOLDS:
            print(f"⚠️ Warning: No threshold defined for module '{module}', using default")
            return 0.75  # Default threshold
        return self.THRESHOLDS[module]
    
    def get_model_for_module(self, module: str) -> str:
        """Get the appropriate model name for a specific module."""
        model_key = self.MODEL_MAPPING.get(module, "default")
        return self.MODEL_CONFIG.get(model_key, self.MODEL_CONFIG.get("default"))
    
    def get_model(self) -> str:
        """Get the default embedding model."""
        return self.EMBEDDING_MODEL
        
    @property
    def config(self):
        """Access to the raw configuration data."""
        return self._config

# Create a singleton instance
config = Config()
