"""
moderator.py
"""

from abc import ABC, abstractmethod


class Moderator(ABC):
    @abstractmethod
    def moderate(self, x: str) -> float:
        pass
