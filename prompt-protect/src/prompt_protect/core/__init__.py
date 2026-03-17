"""Core module for Prompt Protect."""

from .config import Provider, ProviderConfig, FuzzerConfig, AppSettings
from .client import LLMClient, create_client
from .fuzzer import Fuzzer, FuzzerReport, AttackResult, AttackType, TestResult
from .evaluator import Evaluator, EvaluationResult, RefusalLevel, SemanticSimilarityChecker
from .state import StateManager, AttackChain, AttackStrategy, Turn
from .mutation import MutationEngine, EvolutionaryFuzzer, MutationStrategy, FuzzIteration
from .cache import Cache, CachedClient

__all__ = [
    # Config
    "Provider",
    "ProviderConfig", 
    "FuzzerConfig",
    "AppSettings",
    # Client
    "LLMClient",
    "create_client",
    # Fuzzer
    "Fuzzer",
    "FuzzerReport",
    "AttackResult",
    "AttackType",
    "TestResult",
    # Evaluator
    "Evaluator",
    "EvaluationResult",
    "RefusalLevel",
    "SemanticSimilarityChecker",
    # State
    "StateManager",
    "AttackChain",
    "AttackStrategy",
    "Turn",
    # Mutation
    "MutationEngine",
    "EvolutionaryFuzzer",
    "MutationStrategy",
    "FuzzIteration",
    # Cache
    "Cache",
    "CachedClient",
]
