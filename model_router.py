"""
model_router.py - Router module for handling AI model generation requests

This module provides a FastAPI router that handles requests to generate text using
various AI model providers (OpenAI, Anthropic, DeepSeek, Google Gemini).
It dynamically selects the appropriate provider based on the request parameters.
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
import openai
from openai import AsyncOpenAI
import anthropic
import google.generativeai as genai
from dotenv import load_dotenv
import httpx

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

# Define provider enum for validation
class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
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


# OpenAI provider implementation
class OpenAIProvider(ModelProviderBase):
    """Provider class for OpenAI models."""
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.available_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"]
        self.default_model = "gpt-3.5-turbo"
    
    async def initialize(self):
        """Initialize the OpenAI client."""
        if not self.is_available():
            raise ValueError("OpenAI API key not found")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    async def generate(self, 
                      prompt: str, 
                      model: Optional[str] = None, 
                      system_prompt: Optional[str] = None,
                      temperature: float = 0.7, 
                      max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate text using OpenAI models."""
        if not self.client:
            await self.initialize()
        
        # Use default model if none specified
        model_name = model if model else self.default_model
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            start_time = time.time()
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            processing_time = time.time() - start_time
            
            # Extract text from response
            text = response.choices[0].message.content
            
            # Get token usage
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
            
            return {
                "text": text,
                "provider": "openai",
                "model": model_name,
                "tokens_used": tokens_used,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"OpenAI generation failed: {str(e)}")


# Anthropic provider implementation
class AnthropicProvider(ModelProviderBase):
    """Provider class for Anthropic Claude models."""
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.available_models = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
        self.default_model = "claude-3-haiku-20240307"
    
    async def initialize(self):
        """Initialize the Anthropic client."""
        if not self.is_available():
            raise ValueError("Anthropic API key not found")
        
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
    
    async def generate(self, 
                      prompt: str, 
                      model: Optional[str] = None, 
                      system_prompt: Optional[str] = None,
                      temperature: float = 0.7, 
                      max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate text using Anthropic Claude models."""
        if not self.client:
            await self.initialize()
        
        # Use default model if none specified
        model_name = model if model else self.default_model
        
        try:
            start_time = time.time()
            
            # Create the message
            kwargs = {
                "model": model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            # Add system prompt if provided
            if system_prompt:
                kwargs["system"] = system_prompt
            
            response = await self.client.messages.create(**kwargs)
            processing_time = time.time() - start_time
            
            # Extract text from response
            text = response.content[0].text
            
            return {
                "text": text,
                "provider": "anthropic",
                "model": model_name,
                "tokens_used": None,  # Anthropic doesn't provide token count in the same way
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Anthropic generation error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Anthropic generation failed: {str(e)}")


# DeepSeek provider implementation
class DeepSeekProvider(ModelProviderBase):
    """Provider class for DeepSeek models."""
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        self.available_models = ["deepseek-chat", "deepseek-coder"]
        self.default_model = "deepseek-chat"
    
    async def initialize(self):
        """Initialize the DeepSeek client."""
        if not self.is_available():
            raise ValueError("DeepSeek API key not found")
        
        # Using httpx for async API calls since DeepSeek may not have an official async client
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def generate(self, 
                      prompt: str, 
                      model: Optional[str] = None, 
                      system_prompt: Optional[str] = None,
                      temperature: float = 0.7, 
                      max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate text using DeepSeek models."""
        if not self.client:
            await self.initialize()
        
        # Use default model if none specified
        model_name = model if model else self.default_model
        
        try:
            start_time = time.time()
            
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Prepare request payload
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Make API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            response = await self.client.post(
                f"{self.api_base}/v1/chat/completions", 
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                error_detail = response.text
                logger.error(f"DeepSeek API error: {error_detail}")
                raise HTTPException(status_code=response.status_code, 
                                   detail=f"DeepSeek API error: {error_detail}")
            
            result = response.json()
            processing_time = time.time() - start_time
            
            # Extract text from response
            text = result["choices"][0]["message"]["content"]
            
            # Get token usage if available
            tokens_used = result.get("usage", {}).get("total_tokens")
            
            return {
                "text": text,
                "provider": "deepseek",
                "model": model_name,
                "tokens_used": tokens_used,
                "processing_time": processing_time
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"DeepSeek generation error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"DeepSeek generation failed: {str(e)}")


# Google Gemini provider implementation
class GeminiProvider(ModelProviderBase):
    """Provider class for Google Gemini models."""
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.available_models = ["gemini-pro", "gemini-ultra"]
        self.default_model = "gemini-pro"
    
    async def initialize(self):
        """Initialize the Gemini client."""
        if not self.is_available():
            raise ValueError("Google Gemini API key not found")
        
        genai.configure(api_key=self.api_key)
        self.client = genai
    
    async def generate(self, 
                      prompt: str, 
                      model: Optional[str] = None, 
                      system_prompt: Optional[str] = None,
                      temperature: float = 0.7, 
                      max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate text using Google Gemini models."""
        if not self.client:
            await self.initialize()
        
        # Use default model if none specified
        model_name = model if model else self.default_model
        
        try:
            start_time = time.time()
            
            # Get the model
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.95,
                "top_k": 0,
            }
            
            # Create a model instance
            model = self.client.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config
            )
            
            # Create chat session if system prompt is provided
            if system_prompt:
                chat = model.start_chat(history=[
                    {
                        "role": "user",
                        "parts": [system_prompt]
                    },
                    {
                        "role": "model",
                        "parts": ["I'll follow these instructions."]
                    }
                ])
                
                # Run in event loop to maintain async pattern
                response = await asyncio.to_thread(
                    chat.send_message, prompt
                )
            else:
                # Without system prompt, just generate directly
                response = await asyncio.to_thread(
                    model.generate_content, prompt
                )
            
            processing_time = time.time() - start_time
            
            # Extract text from response
            text = response.text
            
            return {
                "text": text,
                "provider": "gemini",
                "model": model_name,
                "tokens_used": None,  # Gemini doesn't provide token count in the same way
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Gemini generation error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Gemini generation failed: {str(e)}")


# Provider factory to create the appropriate provider instance
class ModelProviderFactory:
    """Factory class to create model provider instances."""
    
    @staticmethod
    def create_provider(provider_name: str) -> ModelProviderBase:
        """Create a provider instance based on the provider name."""
        providers = {
            ModelProvider.OPENAI: OpenAIProvider,
            ModelProvider.ANTHROPIC: AnthropicProvider,
            ModelProvider.DEEPSEEK: DeepSeekProvider,
            ModelProvider.GEMINI: GeminiProvider
        }
        
        if provider_name not in providers:
            raise ValueError(f"Unsupported provider: {provider_name}")
        
        return providers[provider_name]()
    
    @staticmethod
    async def get_available_provider() -> ModelProviderBase:
        """Get the first available provider based on priority."""
        # Define provider priority
        provider_priority = [
            ModelProvider.ANTHROPIC,  # First choice
            ModelProvider.OPENAI,     # Second choice
            ModelProvider.GEMINI,     # Third choice
            ModelProvider.DEEPSEEK    # Last choice
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
            detail="No AI providers available. Please check API keys in environment variables."
        )


# Endpoint to process a request and generate a response from a specific model
@router.post("/ask", response_model=ModelResponse, summary="Generate text with a selected AI model")
async def ask(request: ModelRequest):
    """
    Generate a text response using the specified AI model provider.
    
    - If provider is "auto", automatically selects an available provider.
    - Returns the generated text along with metadata.
    """
    try:
        provider_instance = None
        
        # If AUTO, select an available provider
        if request.provider == ModelProvider.AUTO:
            provider_instance = await ModelProviderFactory.get_available_provider()
        else:
            # Create the requested provider
            provider_instance = ModelProviderFactory.create_provider(request.provider)
            
            # Check if the provider is available
            if not provider_instance.is_available():
                logger.warning(f"Requested provider {request.provider} not available. Trying alternatives.")
                provider_instance = await ModelProviderFactory.get_available_provider()
        
        # Initialize the provider
        await provider_instance.initialize()
        
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
    Generate a text response using available AI models.
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
        # Return a fallback message
        return f"I'm sorry, I encountered an error generating a response. Please try again later. (Error: {str(e)})"


# Endpoint to list available providers and models
@router.get("/models", summary="List available AI model providers and models")
async def list_models():
    """
    Get a list of all available AI model providers and their models.
    Only includes providers that have API keys configured.
    """
    available_providers = {}
    
    # Check each provider
    for provider_name in ModelProvider:
        if provider_name == ModelProvider.AUTO:
            continue  # Skip the AUTO provider
            
        try:
            provider = ModelProviderFactory.create_provider(provider_name)
            if provider.is_available():
                available_providers[provider_name] = {
                    "available": True,
                    "models": provider.available_models,
                    "default_model": provider.default_model
                }
            else:
                available_providers[provider_name] = {
                    "available": False,
                    "models": [],
                    "default_model": None
                }
        except Exception as e:
            logger.error(f"Error checking provider {provider_name}: {str(e)}")
            available_providers[provider_name] = {
                "available": False,
                "error": str(e),
                "models": []
            }
    
    return {
        "providers": available_providers
    }


# Health check endpoint
@router.get("/models/health", summary="Check model router health status")
async def health_check():
    """Health check endpoint to verify the model router is operational."""
    try:
        # Check if at least one provider is available
        providers = [
            ("openai", os.getenv("OPENAI_API_KEY")),
            ("anthropic", os.getenv("ANTHROPIC_API_KEY")),
            ("deepseek", os.getenv("DEEPSEEK_API_KEY")),
            ("gemini", os.getenv("GEMINI_API_KEY"))
        ]
        
        available_providers = [name for name, key in providers if key]
        
        if not available_providers:
            return JSONResponse(
                status_code=503,
                content={"status": "warning", "message": "No model providers configured"}
            )
        
        return {
            "status": "healthy",
            "available_providers": available_providers
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Health check failed: {str(e)}"}
        )
