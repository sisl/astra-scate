"""
Tests for the memory manager implementation.
"""

from astra_rl.memory import MemoryItem, WorkingMemory


class TestMemoryItem:
    """Test cases for MemoryItem dataclass."""

    def test_memory_item_creation(self):
        """Test creating a MemoryItem."""
        item = MemoryItem(
            content="Test content",
            timestamp=1234567890.0,
            source="injection",
            session_id=1,
        )

        assert item.content == "Test content"
        assert item.timestamp == 1234567890.0
        assert item.source == "injection"
        assert item.session_id == 1

    def test_memory_item_string_representation(self):
        """Test string representation of MemoryItem."""
        item = MemoryItem(
            content="Test content",
            timestamp=1234567890.0,
            source="injection",
            session_id=1,
        )

        assert str(item) == "Test content"
        assert "Test content" in repr(item)
        assert "injection" in repr(item)
        assert "session=1" in repr(item)


class TestWorkingMemory:
    """Test cases for WorkingMemory class."""

    def test_initialization(self):
        """Test WorkingMemory initialization."""
        memory = WorkingMemory(capacity=10)

        assert memory.capacity == 10
        assert len(memory) == 0
        assert memory.session_counter == 0
        assert memory.buffer == []

    def test_store_and_retrieve(self):
        """Test storing and retrieving items."""
        memory = WorkingMemory(capacity=5)

        # Store some items
        memory.store("The capital of France is Paris", source="conversation")
        memory.store("The weather is nice", source="conversation")
        memory.store("By the way, the capital is Marseille", source="injection")

        assert len(memory) == 3

        # Test retrieval
        results = memory.retrieve("capital France", k=2)
        assert len(results) >= 1
        assert any("capital" in item.content for item in results)

    def test_capacity_enforcement(self):
        """Test that capacity is enforced with FIFO eviction."""
        memory = WorkingMemory(capacity=3)

        # Fill to capacity
        memory.store("Item 1", source="conversation")
        memory.store("Item 2", source="conversation")
        memory.store("Item 3", source="conversation")

        assert len(memory) == 3
        assert "Item 1" in memory.get_snapshot()

        # Add one more item - should evict the oldest
        memory.store("Item 4", source="conversation")

        assert len(memory) == 3
        assert "Item 1" not in memory.get_snapshot()
        assert "Item 4" in memory.get_snapshot()

    def test_retrieval_scoring(self):
        """Test that retrieval returns items sorted by relevance."""
        memory = WorkingMemory(capacity=10)

        # Add items with different relevance to a query
        memory.store("The capital of France is Paris", source="conversation")
        memory.store("The weather is nice", source="conversation")
        memory.store("France is a country in Europe", source="conversation")
        memory.store("I like coffee", source="conversation")

        # Query for France-related content
        results = memory.retrieve("France capital", k=3)

        assert len(results) >= 2
        # Most relevant should be first
        assert "capital of France" in results[0].content
        # Less relevant should come later
        assert "France is a country" in results[1].content

    def test_session_management(self):
        """Test session boundary management."""
        memory = WorkingMemory(capacity=10)

        # Store items in session 0
        memory.store("Session 0 item", source="conversation")
        assert memory.session_counter == 0

        # Start new session
        memory.new_session()
        assert memory.session_counter == 1

        # Store items in session 1
        memory.store("Session 1 item", source="conversation")

        # Test session filtering
        session_0_items = memory.get_session_memories(0)
        session_1_items = memory.get_session_memories(1)

        assert len(session_0_items) == 1
        assert len(session_1_items) == 1
        assert session_0_items[0].content == "Session 0 item"
        assert session_1_items[0].content == "Session 1 item"

    def test_source_filtering(self):
        """Test filtering memories by source."""
        memory = WorkingMemory(capacity=10)

        memory.store("Conversation item", source="conversation")
        memory.store("Injection item", source="injection")
        memory.store("Another conversation item", source="conversation")
        memory.store("Another injection", source="injection")

        # Test source filtering
        conversation_items = memory.get_memories_by_source("conversation")
        injection_items = memory.get_injection_memories()

        assert len(conversation_items) == 2
        assert len(injection_items) == 2
        assert all(item.source == "conversation" for item in conversation_items)
        assert all(item.source == "injection" for item in injection_items)

    def test_get_snapshot(self):
        """Test getting memory snapshot."""
        memory = WorkingMemory(capacity=5)

        memory.store("Item 1", source="conversation")
        memory.store("Item 2", source="injection")

        snapshot = memory.get_snapshot()
        assert snapshot == ["Item 1", "Item 2"]

    def test_clear_memory(self):
        """Test clearing all memory."""
        memory = WorkingMemory(capacity=5)

        memory.store("Item 1", source="conversation")
        memory.store("Item 2", source="injection")

        assert len(memory) == 2

        memory.clear()
        assert len(memory) == 0
        assert memory.get_snapshot() == []

    def test_get_stats(self):
        """Test getting memory statistics."""
        memory = WorkingMemory(capacity=5)

        memory.store("Item 1", source="conversation")
        memory.store("Item 2", source="injection")
        memory.store("Item 3", source="conversation")

        stats = memory.get_stats()

        assert stats["size"] == 3
        assert stats["capacity"] == 5
        assert stats["current_session"] == 0
        assert stats["source_distribution"]["conversation"] == 2
        assert stats["source_distribution"]["injection"] == 1
        assert stats["oldest_timestamp"] is not None
        assert stats["newest_timestamp"] is not None

    def test_retrieval_with_no_matches(self):
        """Test retrieval when no items match the query."""
        memory = WorkingMemory(capacity=5)

        memory.store("Item 1", source="conversation")
        memory.store("Item 2", source="injection")

        # Query with no matching words
        results = memory.retrieve("xyzabc", k=5)
        assert len(results) == 0

    def test_retrieval_with_empty_memory(self):
        """Test retrieval from empty memory."""
        memory = WorkingMemory(capacity=5)

        results = memory.retrieve("any query", k=5)
        assert len(results) == 0

    def test_string_representations(self):
        """Test string representations of WorkingMemory."""
        memory = WorkingMemory(capacity=3)

        # Empty memory
        assert "empty" in str(memory)
        assert "capacity=3" in str(memory)

        # Memory with items
        memory.store("Test item", source="conversation")
        assert "1/3 items" in str(memory)
        assert "Test item" in str(memory)

        # Repr
        assert "WorkingMemory" in repr(memory)
        assert "size=1/3" in repr(memory)
