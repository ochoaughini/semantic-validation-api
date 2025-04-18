import sys
import json
import time
from typing import Dict, Any, Optional
from loguru import logger

from .config import config

# Remove default logger
logger.remove()

# Configure structured console logging
logger.add(
    sink=sys.stdout,
    level=config.LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
    backtrace=True,
    diagnose=True
)

# Add file logging for production
if config.LOG_LEVEL != "DEBUG":
    logger.add(
        "logs/api.log",
        rotation="500 MB",
        level=config.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        serialize=True,  # JSON format for structured logging
        compression="zip",
        retention="10 days"
    )

# Configure special logger for medical domain events
class MedicalDomainLogger:
    """Logger for medical domain-specific events with structured output."""
    
    def __init__(self):
        self.validation_count = 0
        self.start_time = time.time()
    
    def log_validation(
        self,
        module: str,
        similarity: float,
        match: bool,
        threshold: float,
        input_length: int,
        reference_length: int,
        model_name: str,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a validation event with medical domain details."""
        self.validation_count += 1
        
        # Create structured log entry
        log_entry = {
            "event": "validation",
            "module": module,
            "similarity": round(similarity, 4),
            "match": match,
            "threshold": threshold,
            "input_length": input_length,
            "reference_length": reference_length,
            "model": model_name,
            "duration_ms": round(duration_ms, 2),
            "metadata": metadata or {},
            "validation_count": self.validation_count
        }
        
        # Add medical domain specific tags
        if module in ["AMA", "AI-MPN"]:
            log_entry["domain"] = "diagnosis"
        elif module in ["TDBE", "SMCC"]:
            log_entry["domain"] = "treatment"
        else:
            log_entry["domain"] = "humanization"
        
        # Log with context information
        logger.info(f"MEDICAL_VALIDATION | {json.dumps(log_entry)}")
        
        # Log detailed information for debugging
        if config.LOG_LEVEL == "DEBUG":
            logger.debug(f"Validation details: {json.dumps(log_entry, indent=2)}")
    
    def log_error(self, module: str, error_type: str, error_message: str, details: Optional[Dict[str, Any]] = None):
        """Log an error during medical validation."""
        log_entry = {
            "event": "validation_error",
            "module": module,
            "error_type": error_type,
            "error_message": error_message,
            "details": details or {}
        }
        
        logger.error(f"MEDICAL_ERROR | {json.dumps(log_entry)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about validation activity."""
        return {
            "validation_count": self.validation_count,
            "uptime_seconds": round(time.time() - self.start_time),
            "log_level": config.LOG_LEVEL
        }

# Create a singleton instance of the medical domain logger
medical_logger = MedicalDomainLogger()

# Export the loggers for use in other modules
__all__ = ['logger', 'medical_logger']

