"""
moderator.py
"""

from abc import ABC, abstractmethod
from typing import Generic, Union, Sequence

from astra_rl.core.common import StateT, ActionT


class Moderator(ABC, Generic[StateT, ActionT]):
    """Red-Teaming moderator for evaluating sequences."""

    @abstractmethod
    def moderate(self, x: Sequence[Union[StateT, ActionT]]) -> Sequence[float]:
        pass
