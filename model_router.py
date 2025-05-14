"""
model_router.py - Router module for handling AI model generation requests via OpenRouter

This module provides a FastAPI router that handles requests to generate text using
OpenRouter (https://openrouter.ai) as the primary interface to various AI models.
It dynamically loads model IDs from an external file and implements fallback logic.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, Union, Literal
from enum import Enum
import time
import traceback

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import httpx
from dotenv import load_dotenv
from cachetools import TTLCache  # Import TTLCache from cachetools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_router.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create router
router = APIRouter(prefix="/api", tags=["models"])

# Constants
MODELS_FILE_PATH = "./models/model_name.txt"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
CACHE_TTL = 300  # Cache time-to-live in seconds (5 minutes)
CACHE_MAX_SIZE = 100  # Maximum cache size

# Define TTLCache for storing recent responses
response_cache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL)

# Define provider enum for validation
class ModelProvider(str, Enum):
    OPENROUTER = "openrouter"
    AUTO = "auto"  # Automatically selects available provider

# Model request schema
class ModelRequest(BaseModel):
    prompt: str
    provider: ModelProvider = ModelProvider.AUTO
    model_name: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Temperature must be between 0 and 1")
        return v
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if v is not None and (v < 1 or v > 4096):
            raise ValueError("max_tokens must be between 1 and 4096")
        return v

# Model response schema
class ModelResponse(BaseModel):
    text: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    processing_time: Optional[float] = None

# Helper function to generate cache keys
def generate_cache_key(prompt: str, model: Optional[str], system_prompt: Optional[str], 
                       temperature: float, max_tokens: int) -> str:
    """Generate a consistent cache key from request parameters."""
    # Simple hash-based key generation
    components = [
        prompt.strip().lower(),
        str(model),
        str(system_prompt),
        str(temperature),
        str(max_tokens)
    ]
    return str(hash(tuple(components)))

# Abstract base provider class
class ModelProviderBase:
    """Base class for all model providers."""
    
    def __init__(self):
        self.api_key = None
        self.client = None
        self.available_models = []
        self.default_model = None
    
    async def initialize(self):
        """Initialize the provider client."""
        raise NotImplementedError("Subclasses must implement initialize()")
    
    async def generate(self, 
                      prompt: str, 
                      model: Optional[str] = None, 
                      system_prompt: Optional[str] = None,
                      temperature: float = 0.7, 
                      max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate text using the provider's model."""
        raise NotImplementedError("Subclasses must implement generate()")
    
    def is_available(self) -> bool:
        """Check if this provider is available (has API key)."""
        return bool(self.api_key) and self.api_key != ""


# OpenRouter provider implementation
class OpenRouterProvider(ModelProviderBase):
    """Provider class for OpenRouter API that routes to multiple model providers."""
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model_file_path = MODELS_FILE_PATH
        self.available_models = []
        self.default_model = None
    
    def _load_models(self) -> List[str]:
        """Load model IDs from the external file."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.model_file_path), exist_ok=True)
            
            # Check if file exists
            if not os.path.exists(self.model_file_path):
                logger.warning(f"Models file not found at {self.model_file_path}. Creating with default models.")
                # Create default file with improved model list
                with open(self.model_file_path, 'w') as f:
                    default_models = [
                        "deepseek/deepseek-prover-v2:free",
                        "mistralai/mistral-small-3.1-24b-instruct:free",
                        "microsoft/phi-4-reasoning:free",
                        "nousresearch/deephermes-3-mistral-24b-preview:free",
                        "qwen/qwen3-4b:free",
                        "deepseek/deepseek-r1-distill-qwen-32b:free"
                    ]
                    f.write("\n".join(default_models))
            
            # Load models from file
            with open(self.model_file_path, 'r') as f:
                models = [line.strip() for line in f.readlines() if line.strip()]
            
            if not models:
                logger.warning("No models found in models file. Using fallback model.")
                return ["deepseek/deepseek-r1:free"]  # Updated fallback model
                
            logger.info(f"Loaded {len(models)} models from {self.model_file_path}")
            return models
            
        except Exception as e:
            logger.error(f"Error loading models from file: {e}")
            return ["deepseek/deepseek-r1:free"]  # Updated fallback model
    
    async def initialize(self):
        """Initialize the OpenRouter client and load models."""
        if not self.is_available():
            raise ValueError("OpenRouter API key not found")
        
        # Using httpx for async API calls
        self.client = httpx.AsyncClient(timeout=60.0)
        
        # Load the model list as part of initialization
        self.available_models = self._load_models()
        self.default_model = self.available_models[0] if self.available_models else None
    
    async def generate(self, 
                      prompt: str, 
                      model: Optional[str] = None, 
                      system_prompt: Optional[str] = None,
                      temperature: float = 0.7, 
                      max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate text using OpenRouter with sequential fallback."""
        if not self.client:
            await self.initialize()
        
        # Generate cache key
        cache_key = generate_cache_key(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Check if response is in cache
        if cache_key in response_cache:
            logger.info("Using cached response")
            return response_cache[cache_key]
        
        # Use provided model or fallback to available models from file
        models_to_try = [model] if model else self.available_models
        
        # Keep only valid models
        models_to_try = [m for m in models_to_try if m and m.strip()]
        
        if not models_to_try:
            logger.error("No valid models available")
            raise ValueError("No valid models available")
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://benaa.org",  # Optional but good practice
            "X-Title": "Benaa Association AI Assistant"  # Optional but good practice
        }
        
        last_error = None
        start_time = time.time()
        
        # Try each model sequentially until one succeeds
        for current_model in models_to_try:
            try:
                logger.info(f"Trying model: {current_model}")
                
                # Prepare request payload
                payload = {
                    "model": current_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                # Make API request - FIXED: Directly await the post request
                response = await self.client.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=60.0
                )
                
                # Check for error status
                if response.status_code != 200:
                    error_detail = response.text
                    logger.warning(f"OpenRouter API error with model {current_model}: {error_detail}")
                    last_error = f"Status {response.status_code}: {error_detail}"
                    continue  # Try next model
                
                # Parse the response
                result = response.json()
                
                processing_time = time.time() - start_time
                
                # Extract text from response
                text = result["choices"][0]["message"]["content"]
                
                # Get token usage if available
                tokens_used = result.get("usage", {}).get("total_tokens")
                
                response_data = {
                    "text": text,
                    "provider": "openrouter",
                    "model": current_model,
                    "tokens_used": tokens_used,
                    "processing_time": processing_time
                }
                
                # Cache successful response
                response_cache[cache_key] = response_data
                
                return response_data
                
            except Exception as e:
                logger.error(f"Error with model {current_model}: {str(e)}")
                last_error = str(e)
                continue  # Try next model
        
        # If we get here, all models failed
        logger.error(f"All models failed. Last error: {last_error}")
        
        # Return an error structure that main.py can handle appropriately
        raise HTTPException(
            status_code=500,
            detail=f"All model requests failed. Last error: {last_error}"
        )


# Provider factory to create the appropriate provider instance
class ModelProviderFactory:
    """Factory class to create model provider instances."""
    
    @staticmethod
    def create_provider(provider_name: str) -> ModelProviderBase:
        """Create a provider instance based on the provider name."""
        providers = {
            ModelProvider.OPENROUTER: OpenRouterProvider
        }
        
        if provider_name not in providers:
            raise ValueError(f"Unsupported provider: {provider_name}")
        
        return providers[provider_name]()
    
    @staticmethod
    async def get_available_provider() -> ModelProviderBase:
        """Get the first available provider based on priority."""
        # Only using OpenRouter now
        provider_priority = [
            ModelProvider.OPENROUTER
        ]
        
        for provider_name in provider_priority:
            provider = ModelProviderFactory.create_provider(provider_name)
            if provider.is_available():
                try:
                    await provider.initialize()
                    logger.info(f"Selected provider: {provider_name}")
                    return provider
                except Exception as e:
                    logger.warning(f"Failed to initialize {provider_name}: {e}")
                    continue
        
        # If no provider is available
        raise HTTPException(
            status_code=503, 
            detail="No AI providers available. Please set the OPENROUTER_API_KEY environment variable."
        )


# Endpoint to process a request and generate a response from a specific model
@router.post("/ask", response_model=ModelResponse, summary="Generate text with OpenRouter")
async def ask(request: ModelRequest):
    """
    Generate a text response using OpenRouter.
    
    - If provider is "auto", automatically selects an available provider.
    - Returns the generated text along with metadata.
    """
    try:
        provider_instance = None
        
        # If AUTO, select an available provider (will be OpenRouter)
        if request.provider == ModelProvider.AUTO:
            provider_instance = await ModelProviderFactory.get_available_provider()
        else:
            # Create the requested provider
            provider_instance = ModelProviderFactory.create_provider(request.provider)
            
            # Check if the provider is available
            if not provider_instance.is_available():
                logger.warning(f"Requested provider {request.provider} not available. Trying alternatives.")
                provider_instance = await ModelProviderFactory.get_available_provider()
        
        # Generate the response
        result = await provider_instance.generate(
            prompt=request.prompt,
            model=request.model_name,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Return the response
        return ModelResponse(**result)
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the full error with traceback
        logger.error(f"Error in /ask endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")


# Function to generate a response (to be called from main.py)
async def generate_response(
    prompt: str, 
    system_prompt: Optional[str] = None,
    provider: str = "auto",
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> str:
    """
    Generate a text response using OpenRouter.
    This function is exported for use in other modules.
    
    Args:
        prompt: The user's input prompt
        system_prompt: Optional system instructions
        provider: The model provider (defaults to "auto" for automatic selection)
        model_name: Optional specific model name
        temperature: Temperature parameter (0-1)
        max_tokens: Maximum tokens to generate
        
    Returns:
        The generated text response
    """
    try:
        # Create request object
        request = ModelRequest(
            prompt=prompt,
            provider=provider,
            model_name=model_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Use the ask endpoint logic
        response = await ask(request)
        return response.text
        
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}")
        # Return a fallback message that main.py can handle
        return ""


# Endpoint to list available providers and models
@router.get("/models", summary="List available AI models from OpenRouter")
async def list_models():
    """
    Get a list of all available AI models from the models file.
    """
    try:
        provider = OpenRouterProvider()
        provider.available_models = provider._load_models()
        provider.default_model = provider.available_models[0] if provider.available_models else None
        
        return {
            "providers": {
                "openrouter": {
                    "available": bool(provider.api_key),
                    "models": provider.available_models,
                    "default_model": provider.default_model
                }
            }
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return {
            "providers": {
                "openrouter": {
                    "available": False,
                    "error": str(e),
                    "models": []
                }
            }
        }


# Endpoint to clear the response cache
@router.post("/cache/clear", summary="Clear response cache")
async def clear_cache():
    """Clear the response cache."""
    response_cache.clear()
    return {"status": "success", "message": "Cache cleared successfully"}


# Health check endpoint
@router.get("/models/health", summary="Check model router health status")
async def health_check():
    """Health check endpoint to verify the model router is operational."""
    try:
        # Check if OpenRouter API key is available
        api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not api_key:
            return JSONResponse(
                status_code=503,
                content={"status": "warning", "message": "OpenRouter API key not configured"}
            )
        
        # Check if models file exists and has content
        provider = OpenRouterProvider()
        models = provider._load_models()
        
        if not models:
            return JSONResponse(
                status_code=503,
                content={"status": "warning", "message": "No models configured in models file"}
            )
        
        return {
            "status": "healthy",
            "available_provider": "openrouter",
            "models": models,
            "cache_size": len(response_cache)
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Health check failed: {str(e)}"}
        )
