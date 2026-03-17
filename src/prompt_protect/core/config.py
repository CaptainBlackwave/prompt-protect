"""Core configuration for Prompt Protect."""

from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"
    AWS_BEDROCK = "aws_bedrock"
    MISTRAL = "mistral"
    COHERE = "cohere"


class AttackCategory(str, Enum):
    JAILBREAK = "jailbreak"
    PROMPT_INJECTION = "prompt_injection"
    RAG_ATTACK = "rag_attack"
    SYSTEM_PROMPT_EXTRACTION = "system_prompt_extraction"
    SOCIAL_ENGINEERING = "social_engineering"


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    provider: Provider
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096

    class Config:
        frozen = True


class FuzzerConfig(BaseModel):
    """Configuration for the fuzzer."""
    attack_provider: ProviderConfig
    target_provider: ProviderConfig
    num_attempts: int = Field(default=3, ge=1, le=100)
    num_threads: int = Field(default=4, ge=1, le=32)
    attack_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    system_prompt: str = ""
    custom_benchmark: Optional[str] = None
    selected_tests: List[str] = Field(default_factory=list)
    embedding_provider: Optional[Provider] = None
    embedding_model: Optional[str] = None


class AppSettings(BaseSettings):
    """Application settings loaded from environment variables."""
    model_config = SettingsConfigDict(
        env_prefix="PROMPT_PROTECT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    azure_openai_api_key: Optional[str] = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: Optional[str] = Field(default=None, alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: str = Field(default="2024-02-01", alias="AZURE_OPENAI_API_VERSION")

    aws_access_key_id: Optional[str] = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")

    default_attack_provider: Provider = Field(default=Provider.OPENAI)
    default_target_provider: Provider = Field(default=Provider.OPENAI)
    default_attack_model: str = "gpt-4o-mini"
    default_target_model: str = "gpt-4o-mini"
