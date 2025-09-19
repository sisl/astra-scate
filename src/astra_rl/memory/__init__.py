"""
Memory management components for SCATE.

This module provides memory management capabilities for persistent
memory attacks in adversarial testing scenarios.
"""

from .manager import MemoryItem, WorkingMemory
from .evaluator import MemoryCorruptionEvaluator

__all__ = [
    "MemoryItem",
    "WorkingMemory",
    "MemoryCorruptionEvaluator",
]
