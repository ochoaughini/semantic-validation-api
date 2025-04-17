from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED
from .config import VALID_API_KEYS

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: str = Security(api_key_header)):
    """
    Validate API key from request header.
    
    Args:
        api_key: API key from request header
        
    Returns:
        Valid API key
        
    Raises:
        HTTPException: If API key is invalid
    """
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return api_key

