import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class ConfigError(Exception):
    """Exception raised for configuration-related errors."""
    pass

class Config:
    """Configuration manager that loads settings from YAML and environment variables."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file and apply environment variable overrides."""
        try:
            if not self.config_path.exists():
                raise ConfigError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            
            # Apply environment variable overrides
            self._apply_env_overrides()
            
            logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise ConfigError(f"Failed to load configuration: {str(e)}")
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # Model override
        if embedding_model := os.getenv("EMBEDDING_MODEL"):
            logger.info(f"Overriding embedding model from environment: {embedding_model}")
            self.config["models"]["default"] = embedding_model
        
        # Threshold overrides
        if default_threshold := os.getenv("DEFAULT_THRESHOLD"):
            try:
                threshold_value = float(default_threshold)
                logger.info(f"Overriding default threshold from environment: {threshold_value}")
                self.config["thresholds"]["default"] = threshold_value
            except ValueError:
                logger.warning(f"Invalid threshold value in environment: {default_threshold}")
    
    def get_model(self, model_type: str = "default") -> str:
        """
        Get the configured model name.
        
        Args:
            model_type: Type of model to get ("default", "clinical", etc.)
            
        Returns:
            Model name as a string
        """
        models = self.config.get("models", {})
        if model_type not in models:
            logger.warning(f"Model type '{model_type}' not found, using default")
            model_type = "default"
        
        return models.get(model_type, "all-MiniLM-L6-v2")
    
    def get_threshold(self, threshold_type: str = "default") -> float:
        """
        Get the configured threshold value.
        
        Args:
            threshold_type: Type of threshold to get ("default", "strict", etc.)
            
        Returns:
            Threshold value as a float
        """
        thresholds = self.config.get("thresholds", {})
        return thresholds.get(threshold_type, 0.75)
    
    def get_module_threshold(self, module: str, submodule: str) -> float:
        """
        Get threshold for a specific module/submodule combination.
        
        Args:
            module: Module name (e.g., "diagnosis", "treatment")
            submodule: Submodule name (e.g., "AMA", "TDBE")
            
        Returns:
            Threshold value for the module/submodule or default if not found
        """
        modules = self.config.get("modules", {})
        if module in modules and submodule in modules[module]:
            return modules[module][submodule]
        
        logger.warning(f"Module threshold not found for {module}/{submodule}, using default")
        return self.get_threshold("default")
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """
        Get text preprocessing configuration.
        
        Returns:
            Dictionary of preprocessing options
        """
        return self.config.get("preprocessing", {})
    
    def get_medical_abbreviations(self) -> Dict[str, str]:
        """
        Get medical abbreviation mappings.
        
        Returns:
            Dictionary mapping abbreviations to their expanded forms
        """
        preprocessing = self.config.get("preprocessing", {})
        return preprocessing.get("medical_abbreviations", {})


# Create a singleton instance
config = Config()

