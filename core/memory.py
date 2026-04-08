"""Conversation memory management — lightweight, no LangChain dependency."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str
    content: str
    metadata: dict = field(default_factory=dict)


class MemoryManager:
    """Manages conversation history for contextual follow-up questions."""

    def __init__(self) -> None:
        """Initialize conversation memory."""
        self.turns: list[ConversationTurn] = []

    def add_user_message(self, message: str) -> None:
        """Record a user message."""
        self.turns.append(ConversationTurn(role="user", content=message))

    def add_assistant_message(self, message: str, metadata: dict | None = None) -> None:
        """Record an assistant message with optional metadata."""
        self.turns.append(ConversationTurn(role="assistant", content=message, metadata=metadata or {}))

    def get_history_string(self, max_turns: int = 10) -> str:
        """Return recent conversation history as a formatted string."""
        recent = self.turns[-max_turns * 2:] if len(self.turns) > max_turns * 2 else self.turns
        lines: list[str] = []
        for turn in recent:
            prefix = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.content}")
        return "\n".join(lines)

    def get_turns(self) -> list[ConversationTurn]:
        """Return all conversation turns."""
        return list(self.turns)

    def clear(self) -> None:
        """Clear all conversation history."""
        self.turns = []
        logger.info("Conversation memory cleared")
