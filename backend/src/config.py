import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file in development
load_dotenv()

class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass

class Config:
    # Use slots to prevent accidental attribute creation and improve memory usage
    __slots__ = ('BASE_DIR', '_config', 'MODEL_CONFIG', 'MODEL_MAPPING', 
                 'THRESHOLDS', 'MEDICAL_TERMS', 'EMBEDDING_MODEL', 
                 'VALID_API_KEYS', 'FRONTEND_URL', 'ALLOWED_ORIGINS', 'LOG_LEVEL')
    
    # Type hints for class properties
    ALLOWED_ORIGINS: List[str]
    VALID_API_KEYS: Set[str]
    
    # Class constants for default values
    DEFAULT_THRESHOLD = 0.75  # Default similarity threshold
    DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Default embedding model
    VARIATION_THRESHOLD = 0.8  # Threshold for checking variations
    
    def _validate_api_key(self, key: str) -> bool:
        """Validate API key format and minimum requirements."""
        return bool(key and len(key) >= 16 and not key.isspace())
    
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
        
        # Environment variable overrides - with defensive assignment
        try:
            self.EMBEDDING_MODEL = os.getenv(
                "EMBEDDING_MODEL",
                self.MODEL_CONFIG.get("default", self.DEFAULT_MODEL)
            )
            if not self.EMBEDDING_MODEL:
                raise ValueError("Empty model name")
        except Exception as e:
            logger.warning(f"Invalid embedding model configuration: {e}")
            self.EMBEDDING_MODEL = self.DEFAULT_MODEL
        
        # API Keys from environment
        _raw_api_keys = os.getenv("API_KEYS", "")
        # Apply stricter validation for API keys
        self.VALID_API_KEYS = {
            k.strip() for k in _raw_api_keys.split(",")
            if k.strip() and self._validate_api_key(k.strip())
        }
        
        # Add fallback API key if none configured
        if not self.VALID_API_KEYS:
            fallback_key = os.getenv("API_KEY", "dev-api-key-2025")
            if self._validate_api_key(fallback_key):
                self.VALID_API_KEYS = {fallback_key}
                logger.warning("No API keys configured, using fallback development key")
            else:
                logger.warning("No valid API keys found and fallback key is invalid")
                self.VALID_API_KEYS = set()
        
        # Frontend URL for CORS
        self.FRONTEND_URL = os.getenv("FRONTEND_URL")
        
        # Configure CORS origins with security in mind
        if os.getenv("ENVIRONMENT") == "development":
            # For development only - restricted to specific local ports
            self.ALLOWED_ORIGINS = [
                "http://localhost:3000",  # Frontend dev server
                "http://localhost:8080",  # Backend dev server
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8080"
            ]
            logger.warning("⚠️  Running in development mode with localhost origins")
        else:
            # Production - strict CORS
            if self.FRONTEND_URL:
                self.ALLOWED_ORIGINS = [self.FRONTEND_URL]
                # Add render.com domains only for staging environments
                if os.getenv("ENVIRONMENT") == "staging":
                    self.ALLOWED_ORIGINS.append("https://*.onrender.com")
            else:
                # No frontend URL specified - use empty list for maximum security in production
                self.ALLOWED_ORIGINS = []
                logger.warning("⚠️  No FRONTEND_URL configured, CORS will be restrictive")
        
        # Logging configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        
        # Validate environment and configurations
        self._validate_environment()
        
        # Log configuration summary on initialization
        logger.info(f"🔧 Configuration loaded from {os.path.join(self.BASE_DIR, 'config.yaml')}")
        logger.info(f"🧠 Using embedding model: {self.EMBEDDING_MODEL}")
        logger.info(f"🔑 API Keys configured: {len(self.VALID_API_KEYS)}")
        logger.info(f"🌐 CORS allowed origins: {self.ALLOWED_ORIGINS}")
        logger.info(f"📝 Logging level: {self.LOG_LEVEL}")
    
    def _validate_environment(self) -> None:
        """Validate and normalize environment variables."""
        # Validate MODEL_CONFIG
        if not self.MODEL_CONFIG:
            logger.warning("No model configuration found, using defaults")
            self.MODEL_CONFIG = {"default": self.DEFAULT_MODEL}
            
        # Validate THRESHOLDS
        if not self.THRESHOLDS:
            logger.warning("No thresholds configured, using defaults")
            self.THRESHOLDS = {
                "diagnosis": self.DEFAULT_THRESHOLD,
                "treatment": self.DEFAULT_THRESHOLD,
                "humanization": self.DEFAULT_THRESHOLD
            }
        
        # Validate other required configurations
        if not self.EMBEDDING_MODEL:
            logger.warning(f"No embedding model specified, using default: {self.DEFAULT_MODEL}")
            self.EMBEDDING_MODEL = self.DEFAULT_MODEL
    
    def get_medical_abbreviations(self) -> Dict[str, str]:
        """Get dictionary of medical abbreviations and their expansions."""
        try:
            return self.MEDICAL_TERMS.get("abbreviations", {})
        except Exception as e:
            logger.error(f"Error accessing medical abbreviations: {str(e)}")
            return {}
    
    def get_medical_synonyms(self) -> Dict[str, List[str]]:
        """Get dictionary of medical terms and their synonyms."""
        try:
            return self.MEDICAL_TERMS.get("synonyms", {})
        except Exception as e:
            logger.error(f"Error accessing medical synonyms: {str(e)}")
            return {}
    
    def get_threshold_for_module(self, module: str) -> float:
        """Get the validation threshold for a specific module."""
        if module not in self.THRESHOLDS:
            logger.warning(f"No threshold defined for module '{module}', using default {self.DEFAULT_THRESHOLD}")
            return self.DEFAULT_THRESHOLD
        return self.THRESHOLDS[module]
    
    def get_model_for_module(self, module: str) -> str:
        """Get the appropriate model name for a specific module."""
        try:
            model_key = self.MODEL_MAPPING.get(module, "default")
            if model_key not in self.MODEL_CONFIG:
                logger.warning(f"Model key {model_key} not found in config")
                return self.get_model()
            return self.MODEL_CONFIG[model_key]
        except Exception as e:
            logger.error(f"Error getting model for module {module}: {e}")
            return self.get_model()
    
    def get_model(self) -> str:
        """Get the default embedding model."""
        return self.EMBEDDING_MODEL
        
    # We don't provide direct access to the raw config, to ensure configurations
    # are accessed through the proper methods that include validation/defaults

# Create a singleton instance
config = Config()
