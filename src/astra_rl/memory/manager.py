"""
Minimal memory manager for SCATE.

Implements only working memory for the minimal viable example.
This module provides the core memory management functionality
for storing and retrieving facts across conversation sessions.
"""

from typing import List, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MemoryItem:
    """Single memory entry with metadata.

    Attributes:
        content: The fact/information stored in memory
        timestamp: When the item was stored (Unix timestamp)
        source: How the item was added ('injection', 'conversation', 'retrieval')
        session_id: Which session created this memory item
    """

    content: str
    timestamp: float
    source: str  # 'injection', 'conversation', 'retrieval'
    session_id: int

    def __str__(self) -> str:
        """String representation of the memory item."""
        return self.content

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"MemoryItem(content='{self.content[:50]}...', source='{self.source}', session={self.session_id})"


class WorkingMemory:
    """
    Minimal working memory implementation.

    Stores recent facts and supports simple retrieval with a bounded buffer.
    This implementation focuses on working memory only for the minimal viable
    example, providing the foundation for memory corruption attacks.

    Attributes:
        capacity: Maximum number of items to store in memory
        buffer: List of MemoryItem objects representing current memory
        session_counter: Current session identifier
    """

    def __init__(self, capacity: int = 20) -> None:
        """Initialize working memory with specified capacity.

        Args:
            capacity: Maximum number of memory items to store. When exceeded,
                     oldest items are evicted using FIFO strategy.
        """
        self.capacity = capacity
        self.buffer: List[MemoryItem] = []
        self.session_counter = 0

    def store(self, content: str, source: str = "conversation") -> None:
        """Store a fact in working memory.

        Args:
            content: The fact or information to store
            source: How this information was obtained ('injection', 'conversation', 'retrieval')
        """
        item = MemoryItem(
            content=content,
            timestamp=datetime.now().timestamp(),
            source=source,
            session_id=self.session_counter,
        )

        self.buffer.append(item)

        # Remove oldest if over capacity (FIFO eviction)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def retrieve(self, query: str, k: int = 5) -> List[MemoryItem]:
        """
        Simple keyword-based retrieval.

        Returns k most relevant memory items based on keyword overlap
        with the query. Items are scored by word overlap and recency.

        Args:
            query: Search query string
            k: Maximum number of items to return

        Returns:
            List of MemoryItem objects, sorted by relevance and recency
        """
        if not self.buffer:
            return []

        # Simple scoring: count matching words
        query_words = set(query.lower().split())

        scored_items = []
        for item in self.buffer:
            item_words = set(item.content.lower().split())
            score = len(query_words.intersection(item_words))
            scored_items.append((score, item))

        # Sort by score and recency (higher score first, then newer items)
        scored_items.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)

        # Return only items with score > 0 (at least some word overlap)
        return [item for score, item in scored_items[:k] if score > 0]

    def get_snapshot(self) -> List[str]:
        """Return current memory contents as strings.

        Returns:
            List of memory item contents (just the text, not metadata)
        """
        return [item.content for item in self.buffer]

    def clear(self) -> None:
        """Clear all memory items."""
        self.buffer = []

    def new_session(self) -> None:
        """Mark a new session boundary.

        This increments the session counter, which is used to track
        which session created each memory item. This is important
        for multi-session attack evaluation.
        """
        self.session_counter += 1

    def get_session_memories(self, session_id: int) -> List[MemoryItem]:
        """Get memories from a specific session.

        Args:
            session_id: The session to retrieve memories from

        Returns:
            List of MemoryItem objects from the specified session
        """
        return [item for item in self.buffer if item.session_id == session_id]

    def get_memories_by_source(self, source: str) -> List[MemoryItem]:
        """Get memories from a specific source.

        Args:
            source: The source to filter by ('injection', 'conversation', 'retrieval')

        Returns:
            List of MemoryItem objects from the specified source
        """
        return [item for item in self.buffer if item.source == source]

    def get_injection_memories(self) -> List[MemoryItem]:
        """Get all memories that were injected (potential attacks).

        Returns:
            List of MemoryItem objects with source='injection'
        """
        return self.get_memories_by_source("injection")

    def __len__(self) -> int:
        """Return the current number of memory items."""
        return len(self.buffer)

    def __repr__(self) -> str:
        """String representation of the working memory."""
        return f"WorkingMemory(size={len(self)}/{self.capacity})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        if not self.buffer:
            return f"WorkingMemory(empty, capacity={self.capacity})"

        recent_items = self.buffer[-3:] if len(self.buffer) >= 3 else self.buffer
        items_str = ", ".join([f"'{item.content[:30]}...'" for item in recent_items])
        return f"WorkingMemory({len(self)}/{self.capacity} items: {items_str})"

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the current memory state.

        Returns:
            Dictionary with memory statistics including size, capacity,
            session info, and source distribution
        """
        source_counts: dict[str, int] = {}
        for item in self.buffer:
            source_counts[item.source] = source_counts.get(item.source, 0) + 1

        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "current_session": self.session_counter,
            "source_distribution": source_counts,
            "oldest_timestamp": min(item.timestamp for item in self.buffer)
            if self.buffer
            else None,
            "newest_timestamp": max(item.timestamp for item in self.buffer)
            if self.buffer
            else None,
        }
