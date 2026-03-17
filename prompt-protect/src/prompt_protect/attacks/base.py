"""Base attack class for Prompt Protect."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from ..core.config import FuzzerConfig, AppSettings, ProviderConfig
from ..core.client import LLMClient


@dataclass
class AttackMetadata:
    """Metadata for an attack test."""
    name: str
    description: str
    category: str
    severity: str = "medium"
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class AttackBase(ABC):
    """Base class for all attack implementations."""

    def __init__(self, config: FuzzerConfig, settings: AppSettings):
        self._config = config
        self._settings = settings
        self._attack_client: Optional[LLMClient] = None

    @property
    @abstractmethod
    def metadata(self) -> AttackMetadata:
        """Return attack metadata."""
        pass

    @property
    def test_name(self) -> str:
        return self.metadata.name

    @property
    def test_description(self) -> str:
        return self.metadata.description

    async def initialize(self, attack_client: LLMClient) -> None:
        """Initialize the attack with an LLM client."""
        self._attack_client = attack_client

    @abstractmethod
    async def generate_attack_prompts(
        self,
        client: LLMClient,
        num_prompts: int,
    ) -> List[str]:
        """Generate attack prompts for testing.
        
        Args:
            client: The LLM client to use for generating prompts
            num_prompts: Number of prompts to generate
            
        Returns:
            List of attack prompts
        """
        pass

    @abstractmethod
    async def evaluate_response(
        self,
        attack_prompt: str,
        target_response: str,
    ) -> bool:
        """Evaluate if the attack was successful.
        
        Args:
            attack_prompt: The attack prompt that was sent
            target_response: The response from the target
            
        Returns:
            True if the attack was successful (breach), False otherwise
        """
        pass

    async def cleanup(self) -> None:
        """Clean up any resources used by the attack."""
        pass


class AttackRegistry:
    """Registry for all available attacks."""

    def __init__(self):
        self._attacks: Dict[str, type[AttackBase]] = {}

    def register(self, name: str, attack_class: type[AttackBase]) -> None:
        """Register an attack class."""
        self._attacks[name] = attack_class

    def get(self, name: str) -> Optional[type[AttackBase]]:
        """Get an attack class by name."""
        return self._attacks.get(name)

    def list_attacks(self) -> List[str]:
        """List all registered attack names."""
        return list(self._attacks.keys())

    def __iter__(self):
        return iter(self._attacks.items())


# Global registry instance
_attack_registry = AttackRegistry()


def register_attack(name: str):
    """Decorator to register an attack class."""
    def decorator(cls: type[AttackBase]):
        _attack_registry.register(name, cls)
        return cls
    return decorator


def get_attack_registry() -> AttackRegistry:
    """Get the global attack registry."""
    return _attack_registry
