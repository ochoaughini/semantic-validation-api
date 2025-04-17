import time
import psutil
import platform
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends, Request, Query
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..auth import get_api_key
from ..semantic_service import get_quality_metrics
from ..logging_config import logger, medical_logger

# Create router with prefix and tags
router = APIRouter(
    prefix="/api/metrics",
    tags=["metrics"],
    dependencies=[Depends(get_api_key)]
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# --- Models --- #

class ModuleMetrics(BaseModel):
    """Metrics for a specific module."""
    accuracy: float = Field(..., description="Accuracy score (0-1)")
    avg_time_ms: float = Field(..., description="Average processing time in milliseconds")
    attempts: int = Field(..., description="Total number of validation attempts")
    successful: int = Field(..., description="Number of successful validations")
    errors: int = Field(..., description="Number of errors encountered")

class DomainMetrics(BaseModel):
    """Metrics for a domain (diagnosis, treatment, humanization)."""
    accuracy: float = Field(..., description="Overall accuracy for the domain")

class OverallMetrics(BaseModel):
    """Overall API metrics."""
    total_validations: int = Field(..., description="Total number of validations")
    error_rate: float = Field(..., description="Error rate (0-1)")
    success_rate: float = Field(..., description="Success rate (0-1)")
    models_loaded: int = Field(..., description="Number of models loaded")
    uptime_stats: Dict[str, Any] = Field(..., description="Uptime statistics")

class QualityMetricsResponse(BaseModel):
    """Response model for quality metrics."""
    modules: Dict[str, ModuleMetrics] = Field(..., description="Metrics per module")
    domains: Dict[str, DomainMetrics] = Field(..., description="Metrics per domain")
    overall: OverallMetrics = Field(..., description="Overall metrics")

class SystemResourcesResponse(BaseModel):
    """Response model for system resources."""
    cpu_usage: float = Field(..., description="CPU usage (%)")
    memory_usage: Dict[str, Any] = Field(..., description="Memory usage")
    disk_usage: Dict[str, Any] = Field(..., description="Disk usage")
    python_version: str = Field(..., description="Python version")
    platform_info: str = Field(..., description="Platform information")
    process_info: Dict[str, Any] = Field(..., description="Process information")

# --- Helper Functions --- #

def get_system_resources() -> Dict[str, Any]:
    """
    Get system resource usage statistics.
    
    Returns:
        Dictionary of system resource metrics
    """
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.5)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent": memory.percent
        }
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = {
            "total_gb": round(disk.total / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "percent": disk.percent
        }
        
        # Process information
        process = psutil.Process()
        process_info = {
            "cpu_percent": process.cpu_percent(interval=0.5),
            "memory_percent": process.memory_percent(),
            "threads": process.num_threads(),
            "created_time": process.create_time()
        }
        
        return {
            "cpu_usage": cpu_percent,
            "memory_usage": memory_usage,
            "disk_usage": disk_usage,
            "python_version": platform.python_version(),
            "platform_info": platform.platform(),
            "process_info": process_info
        }
    except Exception as e:
        logger.error(f"Error collecting system resources: {str(e)}")
        return {
            "error": str(e),
            "cpu_usage": 0.0,
            "memory_usage": {},
            "disk_usage": {},
            "python_version": platform.python_version(),
            "platform_info": platform.platform(),
            "process_info": {}
        }

# --- Routes --- #

@router.get("/quality", response_model=QualityMetricsResponse)
@limiter.limit("20/minute")
async def get_quality_metrics_endpoint(
    req: Request,
    module: Optional[str] = Query(None, description="Filter metrics by module")
):
    """
    Get quality metrics for the semantic validation service.
    
    Args:
        module: Optional filter for a specific module
        
    Returns:
        Dictionary of quality metrics
    """
    try:
        # Get metrics from semantic service
        metrics = get_quality_metrics()
        
        # Filter by module if requested
        if module and module in metrics["modules"]:
            filtered_modules = {module: metrics["modules"][module]}
            metrics["modules"] = filtered_modules
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error retrieving quality metrics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve quality metrics"
        )

@router.get("/system", response_model=SystemResourcesResponse)
@limiter.limit("10/minute")
async def get_system_resources_endpoint(req: Request):
    """
    Get system resource usage statistics.
    
    Returns:
        Dictionary of system resource metrics
    """
    try:
        return get_system_resources()
    
    except Exception as e:
        logger.error(f"Error retrieving system resources: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve system resources"
        )

@router.get("/health")
@limiter.limit("100/minute")
async def get_health_metrics(req: Request):
    """
    Get detailed health metrics for the service.
    
    Returns:
        Health status of various components
    """
    start_time = time.time()
    
    try:
        # Check database connectivity (placeholder)
        db_status = "ok"
        
        # Check model availability
        model_metrics = get_quality_metrics()
        model_status = "ok" if model_metrics["overall"]["models_loaded"] > 0 else "error"
        
        # Check system resources
        system_resources = get_system_resources()
        system_status = "warning" if system_resources["cpu_usage"] > 80 else "ok"
        
        # Calculate response time
        response_time = (time.time() - start_time) * 1000
        
        return {
            "status": "ok",
            "components": {
                "database": db_status,
                "models": model_status,
                "system": system_status
            },
            "metrics": {
                "uptime": model_metrics["overall"]["uptime_stats"],
                "cpu_usage": system_resources["cpu_usage"],
                "memory_usage": system_resources["memory_usage"]["percent"],
                "response_time_ms": round(response_time, 2)
            },
            "version": "1.0.0"
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "response_time_ms": round((time.time() - start_time) * 1000, 2)
        }

