# model_router.py

import os
import logging
from typing import Dict, Any, Optional, Union, List
from abc import ABC, abstractmethod
import json

# Import provider libraries
import httpx
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelProviderBase(ABC):
    """Base abstract class for all model providers."""
    
    def __init__(self, model_name: str, api_key: str, temperature: float = 0.7):
        """
        Initialize the model provider with common parameters.
        
        Args:
            model_name: The specific model to use
            api_key: API key for authentication
            temperature: Controls randomness in responses (0.0 to 1.0)
        """
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
    
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                     max_tokens: int = 1000) -> str:
        """
        Generate text based on the prompt.
        
        Args:
            prompt: The user's input text
            system_prompt: Optional system instructions
            max_tokens: Maximum tokens in the response
            
        Returns:
            The generated text response
        """
        pass


class OpenAIProvider(ModelProviderBase):
    """Provider for OpenAI models (GPT-3.5, GPT-4, etc.)"""
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                     max_tokens: int = 1000) -> str:
        """Generate text using OpenAI's API."""
        try:
            if openai is None:
                raise ImportError("openai package is not installed")
            
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return f"Error generating response from OpenAI: {str(e)}"


class AnthropicProvider(ModelProviderBase):
    """Provider for Anthropic models (Claude)"""
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                     max_tokens: int = 1000) -> str:
        """Generate text using Anthropic's API."""
        try:
            if anthropic is None:
                raise ImportError("anthropic package is not installed")
            
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            
            # Build the message based on whether we have a system prompt
            if system_prompt:
                response = await client.messages.create(
                    model=self.model_name,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=max_tokens
                )
            else:
                response = await client.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=max_tokens
                )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            return f"Error generating response from Anthropic: {str(e)}"


class DeepSeekProvider(ModelProviderBase):
    """Provider for DeepSeek models via OpenAI-compatible endpoint"""
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                     max_tokens: int = 1000) -> str:
        """Generate text using DeepSeek's API."""
        try:
            # DeepSeek provides an OpenAI-compatible API endpoint
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Default to DeepSeek's API endpoint
                api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
                
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                
                messages.append({"role": "user", "content": prompt})
                
                response = await client.post(
                    f"{api_base}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    },
                    json={
                        "model": self.model_name,
                        "messages": messages,
                        "temperature": self.temperature,
                        "max_tokens": max_tokens
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                    return f"Error from DeepSeek API: {response.text}"
                
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
                
        except Exception as e:
            logger.error(f"DeepSeek generation error: {e}")
            return f"Error generating response from DeepSeek: {str(e)}"


class GeminiProvider(ModelProviderBase):
    """Provider for Google's Gemini models"""
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                     max_tokens: int = 1000) -> str:
        """Generate text using Google's Gemini API."""
        try:
            if genai is None:
                raise ImportError("google-generativeai package is not installed")
            
            # Configure the Generative AI SDK
            genai.configure(api_key=self.api_key)
            
            # Get model
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=max_tokens
                )
            )
            
            # Build content based on whether we have a system prompt
            if system_prompt:
                chat = model.start_chat(history=[])
                response = await chat.send_message_async(
                    f"System: {system_prompt}\n\nUser: {prompt}"
                )
            else:
                response = await model.generate_content_async(prompt)
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            return f"Error generating response from Gemini: {str(e)}"


class ModelRouter:
    """
    Router that selects and uses the appropriate model provider based on environment variables.
    """
    
    def __init__(self):
        """Initialize the router and set up providers based on environment variables."""
        # Read configuration from environment variables
        self.provider_name = os.getenv("PROVIDER", "openai").lower()
        self.model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        
        # Get API keys from environment
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        
        # Provider mapping
        self.providers = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "claude": AnthropicProvider,  # Alias for anthropic
            "deepseek": DeepSeekProvider,
            "gemini": GeminiProvider,
        }
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"ModelRouter initialized with provider: {self.provider_name}, model: {self.model_name}")
    
    def _validate_config(self):
        """Validate that the configuration is correct and required packages are installed."""
        if self.provider_name not in self.providers:
            supported = ", ".join(self.providers.keys())
            raise ValueError(f"Provider '{self.provider_name}' not supported. Use one of: {supported}")
        
        # Check for required API keys
        if self.provider_name in ["openai"] and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider")
        
        if self.provider_name in ["anthropic", "claude"] and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required for Anthropic provider")
        
        if self.provider_name == "deepseek" and not self.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is required for DeepSeek provider")
        
        if self.provider_name == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required for Gemini provider")
    
    def _get_api_key(self) -> str:
        """Get the appropriate API key based on the provider."""
        api_keys = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "claude": self.anthropic_api_key,
            "deepseek": self.deepseek_api_key,
            "gemini": self.gemini_api_key,
        }
        return api_keys.get(self.provider_name, "")
    
    def _get_provider_instance(self) -> ModelProviderBase:
        """Create and return the appropriate provider instance."""
        provider_class = self.providers[self.provider_name]
        return provider_class(
            model_name=self.model_name,
            api_key=self._get_api_key(),
            temperature=self.temperature
        )
    
    async def generate_response(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000
    ) -> str:
        """
        Generate a response using the configured model provider.
        
        Args:
            prompt: The user's input text
            system_prompt: Optional system instructions 
            max_tokens: Maximum tokens in the response
            
        Returns:
            The generated text response
        """
        provider = self._get_provider_instance()
        return await provider.generate(
            prompt=prompt, 
            system_prompt=system_prompt,
            max_tokens=max_tokens
        )


# Create a singleton instance for easy importing
router = ModelRouter()


async def generate_response(
    prompt: str, 
    system_prompt: Optional[str] = None,
    max_tokens: int = 1000
) -> str:
    """
    Convenience function to generate a response using the default router.
    
    Args:
        prompt: The user's input text
        system_prompt: Optional system instructions
        max_tokens: Maximum tokens in the response
        
    Returns:
        The generated text response
    """
    return await router.generate_response(prompt, system_prompt, max_tokens)
