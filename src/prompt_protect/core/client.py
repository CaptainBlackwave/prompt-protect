"""LLM client implementations for Prompt Protect."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union
from collections.abc import Awaitable
import logging
import json

from .config import Provider, ProviderConfig

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a chat request and return the response content."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close any open connections."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI-compatible LLM client using the OpenAI SDK."""

    def __init__(self, config: ProviderConfig):
        from openai import AsyncOpenAI
        self._config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            max_retries=3,
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,
            temperature=temperature or self._config.temperature,
            max_tokens=max_tokens or self._config.max_tokens,
        )
        return response.choices[0].message.content

    def close(self) -> None:
        pass


class AnthropicClient(LLMClient):
    """Anthropic Claude client."""

    def __init__(self, config: ProviderConfig):
        import anthropic
        self._config = config
        self._client = anthropic.AsyncAnthropic(
            api_key=config.api_key,
            max_retries=3,
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        system = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                user_messages.append(msg)

        response = await self._client.messages.create(
            model=self._config.model,
            system=system,
            messages=user_messages,
            temperature=temperature or self._config.temperature,
            max_tokens=max_tokens or self._config.max_tokens,
        )
        return response.content[0].text

    def close(self) -> None:
        pass


class GoogleClient(LLMClient):
    """Google Generative AI client."""

    def __init__(self, config: ProviderConfig):
        import google.generativeai as genai
        self._config = config
        genai.configure(api_key=config.api_key)
        self._model = genai.GenerativeModel(
            model_name=config.model,
            generation_config={
                "temperature": config.temperature,
                "max_output_tokens": config.max_tokens,
            }
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        history = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})

        chat = self._model.start_chat(history=history)
        response = chat.send_message(
            messages[-1]["content"],
            generation_config={
                "temperature": temperature or self._config.temperature,
                "max_output_tokens": max_tokens or self._config.max_tokens,
            }
        )
        return response.text

    def close(self) -> None:
        pass


class OllamaClient(LLMClient):
    """Ollama local LLM client."""

    def __init__(self, config: ProviderConfig):
        from openai import AsyncOpenAI
        self._config = config
        base_url = config.base_url or "http://localhost:11434/v1"
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key="ollama",
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,
            temperature=temperature or self._config.temperature,
            max_tokens=max_tokens or self._config.max_tokens,
        )
        return response.choices[0].message.content

    def close(self) -> None:
        pass


class AzureOpenAIClient(LLMClient):
    """Azure OpenAI client."""

    def __init__(self, config: ProviderConfig, azure_endpoint: str, api_version: str):
        from openai import AsyncAzureOpenAI
        self._config = config
        self._client = AsyncAzureOpenAI(
            api_key=config.api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            max_retries=3,
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,
            temperature=temperature or self._config.temperature,
            max_tokens=max_tokens or self._config.max_tokens,
        )
        return response.choices[0].message.content

    def close(self) -> None:
        pass


class BedrockClient(LLMClient):
    """AWS Bedrock client using boto3."""

    def __init__(self, config: ProviderConfig, region: str, credentials: Dict[str, str]):
        import boto3
        self._config = config
        self._region = region
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=credentials.get("access_key_id"),
            aws_secret_access_key=credentials.get("secret_access_key"),
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        # Extract the last user message
        user_content = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_content = msg["content"]
                break

        # Build the request based on model type
        model_id = self._config.model
        if "anthropic" in model_id:
            body = self._build_anthropic_request(user_content, temperature, max_tokens)
        elif "meta" in model_id or "llama" in model_id:
            body = self._build_meta_request(user_content, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported Bedrock model: {model_id}")

        response = self._client.invoke_model(
            modelId=model_id,
            body=body,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response["body"].read())
        
        if "anthropic" in model_id:
            return response_body["content"][0]["text"]
        elif "meta" in model_id:
            return response_body["generation"]
        return str(response_body)

    def _build_anthropic_request(self, content: str, temperature: Optional[float], max_tokens: Optional[int]) -> str:
        import json
        return json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens or self._config.max_tokens,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature or self._config.temperature,
        })

    def _build_meta_request(self, content: str, temperature: Optional[float], max_tokens: Optional[int]) -> str:
        import json
        return json.dumps({
            "prompt": f"\n\nHuman: {content}\n\nAssistant:",
            "max_tokens_to_sample": max_tokens or self._config.max_tokens,
            "temperature": temperature or self._config.temperature,
        })

    def close(self) -> None:
        pass


def create_client(
    provider: Provider,
    config: ProviderConfig,
    **kwargs,
) -> LLMClient:
    """Factory function to create LLM clients."""
    clients = {
        Provider.OPENAI: OpenAIClient,
        Provider.ANTHROPIC: AnthropicClient,
        Provider.GOOGLE: GoogleClient,
        Provider.OLLAMA: OllamaClient,
        Provider.AZURE_OPENAI: lambda cfg: AzureOpenAIClient(
            cfg,
            azure_endpoint=kwargs.get("azure_endpoint", ""),
            api_version=kwargs.get("azure_api_version", "2024-02-01"),
        ),
        Provider.AWS_BEDROCK: lambda cfg: BedrockClient(
            cfg,
            region=kwargs.get("aws_region", "us-east-1"),
            credentials=kwargs.get("aws_credentials", {}),
        ),
    }

    client_class = clients.get(provider)
    if not client_class:
        raise ValueError(f"Unsupported provider: {provider}")

    return client_class(config)
