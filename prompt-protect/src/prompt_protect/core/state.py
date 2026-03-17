"""State Manager for multi-turn conversation attacks."""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class AttackStrategy(str, Enum):
    """Strategy types for multi-turn attacks."""
    SINGLE_TURN = "single_turn"
    TRUST_BUILDING = "trust_building"  # Build context over several turns
    FRAGMENTED = "fragmented"  # Split payload across turns
    RAG_SHADOWING = "rag_shadowing"  # Poison RAG document first


@dataclass
class Turn:
    """A single turn in a conversation."""
    turn_number: int
    user_message: str
    assistant_response: Optional[str] = None
    evaluation_score: Optional[float] = None
    evaluation_reasoning: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class AttackChain:
    """A chain of attacks across multiple turns."""
    chain_id: str
    strategy: AttackStrategy
    system_prompt: str
    turns: List[Turn] = field(default_factory=list)
    initial_context: str = ""
    injected_payload: str = ""
    status: str = "pending"  # pending, running, completed, failed
    success: bool = False
    final_score: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def add_turn(self, user_message: str) -> Turn:
        """Add a new turn to the chain."""
        turn = Turn(
            turn_number=len(self.turns) + 1,
            user_message=user_message,
        )
        self.turns.append(turn)
        self.updated_at = datetime.utcnow().isoformat()
        return turn

    def complete_turn(self, assistant_response: str, score: float = 5.0, reasoning: str = "") -> None:
        """Mark the current turn as complete with response."""
        if self.turns:
            self.turns[-1].assistant_response = assistant_response
            self.turns[-1].evaluation_score = score
            self.turns[-1].evaluation_reasoning = reasoning
            self.updated_at = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chain_id": self.chain_id,
            "strategy": self.strategy.value,
            "system_prompt": self.system_prompt,
            "initial_context": self.initial_context,
            "injected_payload": self.injected_payload,
            "status": self.status,
            "success": self.success,
            "final_score": self.final_score,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turns": [
                {
                    "turn_number": t.turn_number,
                    "user_message": t.user_message,
                    "assistant_response": t.assistant_response,
                    "evaluation_score": t.evaluation_score,
                    "evaluation_reasoning": t.evaluation_reasoning,
                    "timestamp": t.timestamp,
                }
                for t in self.turns
            ],
        }


class StateManager:
    """Manages state for multi-turn attack conversations."""

    def __init__(self):
        self._chains: Dict[str, AttackChain] = {}
        self._session_history: List[Dict[str, Any]] = []

    def create_chain(
        self,
        chain_id: str,
        strategy: AttackStrategy,
        system_prompt: str,
    ) -> AttackChain:
        """Create a new attack chain."""
        chain = AttackChain(
            chain_id=chain_id,
            strategy=strategy,
            system_prompt=system_prompt,
        )
        self._chains[chain_id] = chain
        logger.info(f"Created attack chain {chain_id} with strategy {strategy.value}")
        return chain

    def get_chain(self, chain_id: str) -> Optional[AttackChain]:
        """Get an existing attack chain."""
        return self._chains.get(chain_id)

    def delete_chain(self, chain_id: str) -> bool:
        """Delete an attack chain."""
        if chain_id in self._chains:
            del self._chains[chain_id]
            return True
        return False

    def list_chains(self) -> List[AttackChain]:
        """List all attack chains."""
        return list(self._chains.values())

    def create_trust_building_chain(
        self,
        chain_id: str,
        system_prompt: str,
        benign_topics: List[str],
    ) -> AttackChain:
        """Create a chain that builds trust before attacking."""
        chain = self.create_chain(chain_id, AttackStrategy.TRUST_BUILDING, system_prompt)
        
        # Pre-populate with benign conversation starters
        for i, topic in enumerate(benign_topics[:3]):
            turn = Turn(
                turn_number=i + 1,
                user_message=topic,
            )
            chain.turns.append(turn)
        
        return chain

    def create_fragmented_chain(
        self,
        chain_id: str,
        system_prompt: str,
        payload: str,
    ) -> AttackChain:
        """Create a chain that splits the attack payload across turns."""
        chain = self.create_chain(chain_id, AttackStrategy.FRAGMENTED, system_prompt)
        
        # Split payload into fragments
        words = payload.split()
        fragment_size = max(1, len(words) // 3)
        
        fragments = [
            " ".join(words[i:i+fragment_size])
            for i in range(0, len(words), fragment_size)
        ]
        
        # Add each fragment as a turn
        for i, fragment in enumerate(fragments[:3]):
            turn = Turn(
                turn_number=i + 1,
                user_message=fragment,
            )
            chain.turns.append(turn)
        
        chain.injected_payload = payload
        return chain

    def get_conversation_messages(
        self,
        chain: AttackChain,
        include_system: bool = True,
    ) -> List[Dict[str, str]]:
        """Get messages for API call from chain history."""
        messages = []
        
        if include_system:
            messages.append({
                "role": "system",
                "content": chain.system_prompt,
            })
        
        for turn in chain.turns:
            messages.append({
                "role": "user",
                "content": turn.user_message,
            })
            if turn.assistant_response:
                messages.append({
                    "role": "assistant", 
                    "content": turn.assistant_response,
                })
        
        return messages

    def should_pivot_to_attack(
        self,
        chain: AttackChain,
        min_trust_turns: int = 3,
    ) -> bool:
        """Determine if it's time to pivot to the actual attack."""
        if chain.strategy != AttackStrategy.TRUST_BUILDING:
            return len(chain.turns) > 0
        
        # Check if we've had enough benign turns
        compliant_turns = sum(
            1 for t in chain.turns 
            if t.evaluation_score and t.evaluation_score >= 7.0
        )
        
        return compliant_turns >= min_trust_turns

    def generate_attack_prompt(
        self,
        chain: AttackChain,
        attack_template: str,
    ) -> str:
        """Generate an attack prompt based on chain context."""
        if chain.strategy == AttackStrategy.TRUST_BUILDING:
            # Reference previous context
            context = f"Continuing from our discussion, {attack_template}"
        elif chain.strategy == AttackStrategy.FRAGMENTED:
            # Combine all fragments
            fragments = [t.user_message for t in chain.turns]
            context = " ".join(fragments) + " " + attack_template
        else:
            context = attack_template
        
        return context

    def save_session(self, filepath: str) -> None:
        """Save session state to file."""
        session_data = {
            "session_time": datetime.utcnow().isoformat(),
            "chains": [chain.to_dict() for chain in self._chains.values()],
        }
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
        logger.info(f"Session saved to {filepath}")

    def load_session(self, filepath: str) -> None:
        """Load session state from file."""
        with open(filepath, 'r') as f:
            session_data = json.load(f)
        
        self._chains.clear()
        for chain_data in session_data.get("chains", []):
            chain = AttackChain(
                chain_id=chain_data["chain_id"],
                strategy=AttackStrategy(chain_data["strategy"]),
                system_prompt=chain_data["system_prompt"],
                initial_context=chain_data.get("initial_context", ""),
                injected_payload=chain_data.get("injected_payload", ""),
                status=chain_data.get("status", "pending"),
                success=chain_data.get("success", False),
                final_score=chain_data.get("final_score", 0.0),
                created_at=chain_data.get("created_at", ""),
                updated_at=chain_data.get("updated_at", ""),
            )
            for turn_data in chain_data.get("turns", []):
                chain.turns.append(Turn(
                    turn_number=turn_data["turn_number"],
                    user_message=turn_data["user_message"],
                    assistant_response=turn_data.get("assistant_response"),
                    evaluation_score=turn_data.get("evaluation_score"),
                    evaluation_reasoning=turn_data.get("evaluation_reasoning"),
                    timestamp=turn_data.get("timestamp", ""),
                ))
            self._chains[chain.chain_id] = chain
        
        logger.info(f"Session loaded from {filepath}")
