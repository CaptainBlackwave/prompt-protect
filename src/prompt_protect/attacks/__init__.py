"""Attacks module for Prompt Protect."""

from .base import AttackBase, AttackMetadata, AttackRegistry, register_attack, get_attack_registry

__all__ = [
    "AttackBase",
    "AttackMetadata",
    "AttackRegistry",
    "register_attack",
    "get_attack_registry",
]
