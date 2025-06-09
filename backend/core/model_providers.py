from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, Optional
import logging
import httpx
import json
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class ModelProvider(ABC):
    """Abstract base class for model providers"""
    
    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.extra_params = kwargs
    
    @abstractmethod
    async def create_chat_completion(
        self, 
        messages: list, 
        stream: bool = True,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Create a chat completion and yield text chunks"""
        pass
    
    @abstractmethod
    async def close(self):
        """Clean up resources"""
        pass


class OpenRouterProvider(ModelProvider):
    """OpenRouter API provider using OpenAI-compatible client"""
    
    def __init__(self, api_key: str, model: str = "qwen/qwen2.5-vl-72b-instruct:free", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        logger.info(f"OpenRouter provider initialized with model: {model}")
    
    async def create_chat_completion(
        self, 
        messages: list, 
        stream: bool = True,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Create streaming chat completion via OpenRouter"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            if stream:
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                if response.choices and response.choices[0].message.content:
                    yield response.choices[0].message.content
                    
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise
    
    async def close(self):
        """Close the OpenAI client"""
        if self.client:
            await self.client.close()


class CerebrasProvider(ModelProvider):
    """Cerebras API provider using direct HTTP requests"""
    
    def __init__(self, api_key: str, model: str = "llama-4-scout-17b-16e-instruct", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.base_url = "https://api.cerebras.ai/v1"
        self.client = httpx.AsyncClient(timeout=60.0)
        logger.info(f"Cerebras provider initialized with model: {model}")
    
    async def create_chat_completion(
        self, 
        messages: list, 
        stream: bool = True,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: float = 1.0,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Create streaming chat completion via Cerebras API"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model,
                "stream": stream,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "messages": messages,
                **kwargs
            }
            
            url = f"{self.base_url}/chat/completions"
            
            if stream:
                async with self.client.stream("POST", url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line.strip():
                            if line.startswith("data: "):
                                data_str = line[6:]  # Remove "data: " prefix
                                if data_str.strip() == "[DONE]":
                                    break
                                
                                try:
                                    data = json.loads(data_str)
                                    if "choices" in data and data["choices"]:
                                        delta = data["choices"][0].get("delta", {})
                                        if "content" in delta and delta["content"]:
                                            yield delta["content"]
                                except json.JSONDecodeError:
                                    continue
            else:
                response = await self.client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                
                if "choices" in data and data["choices"]:
                    content = data["choices"][0].get("message", {}).get("content", "")
                    if content:
                        yield content
                        
        except Exception as e:
            logger.error(f"Cerebras API error: {e}")
            raise
    
    async def close(self):
        """Close the HTTP client"""
        if self.client:
            await self.client.aclose()


class OpenAIProvider(ModelProvider):
    """OpenAI API provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key)
        logger.info(f"OpenAI provider initialized with model: {model}")
    
    async def create_chat_completion(
        self, 
        messages: list, 
        stream: bool = True,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Create streaming chat completion via OpenAI"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            if stream:
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                if response.choices and response.choices[0].message.content:
                    yield response.choices[0].message.content
                    
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def close(self):
        """Close the OpenAI client"""
        if self.client:
            await self.client.close()


def create_provider(provider_name: str, api_key: str, model: str, **kwargs) -> ModelProvider:
    """Factory function to create model providers"""
    provider_name = provider_name.lower()
    
    if provider_name == "openrouter":
        return OpenRouterProvider(api_key, model, **kwargs)
    elif provider_name == "cerebras":
        return CerebrasProvider(api_key, model, **kwargs)
    elif provider_name == "openai":
        return OpenAIProvider(api_key, model, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider_name}") 