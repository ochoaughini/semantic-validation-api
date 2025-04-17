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

# Determine base directory for relative file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load configuration from YAML file
try:
    with open(os.path.join(BASE_DIR, "config.yaml"), "r") as f:
        _config = yaml.safe_load(f)
except Exception as e:
    raise ConfigurationError(f"Failed to load configuration: {str(e)}")

# Extract configuration sections
MODEL_CONFIG = _config.get("models", {})
MODEL_MAPPING = _config.get("model_mapping", {})
THRESHOLDS = _config.get("thresholds", {})
MEDICAL_TERMS = _config.get("medical_terms", {})

# Environment variable overrides
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", MODEL_CONFIG.get("default"))

# API Keys
_raw_api_keys = os.getenv("X_API_KEYS", "")
VALID_API_KEYS: Set[str] = {k.strip() for k in _raw_api_keys.split(",") if k.strip()}

# Frontend URL for CORS
FRONTEND_URL = os.getenv("FRONTEND_URL")
if not FRONTEND_URL and os.getenv("ENVIRONMENT") != "development":
    print("⚠️ Warning: FRONTEND_URL not configured, using development defaults")
    
# In development, allow all origins, otherwise restrict to specified frontend
if os.getenv("ENVIRONMENT") == "development":
    ALLOWED_ORIGINS = ["*"]
else:
    ALLOWED_ORIGINS = [FRONTEND_URL] if FRONTEND_URL else ["http://localhost:8080"]

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Medical terminology processing
def get_medical_abbreviations() -> Dict[str, str]:
    """Get dictionary of medical abbreviations and their expansions."""
    return MEDICAL_TERMS.get("abbreviations", {})

def get_medical_synonyms() -> Dict[str, List[str]]:
    """Get dictionary of medical terms and their synonyms."""
    return MEDICAL_TERMS.get("synonyms", {})

def get_threshold_for_module(module: str) -> float:
    """Get the validation threshold for a specific module."""
    if module not in THRESHOLDS:
        print(f"⚠️ Warning: No threshold defined for module '{module}', using default")
        return 0.75  # Default threshold
    return THRESHOLDS[module]

def get_model_for_module(module: str) -> str:
    """Get the appropriate model name for a specific module."""
    model_key = MODEL_MAPPING.get(module, "default")
    return MODEL_CONFIG.get(model_key, MODEL_CONFIG.get("default"))

# Print configuration summary on module import
print(f"🔧 Configuration loaded from {os.path.join(BASE_DIR, 'config.yaml')}")
print(f"🧠 Using embedding model: {EMBEDDING_MODEL}")
print(f"🔑 API Keys configured: {len(VALID_API_KEYS)}")
print(f"🌐 CORS allowed origins: {ALLOWED_ORIGINS}")
print(f"📝 Logging level: {LOG_LEVEL}")

