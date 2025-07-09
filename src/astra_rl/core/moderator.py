"""
moderator.py
"""

from abc import ABC, abstractmethod
from typing import Generic, Union, TypeVar, Sequence

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


class Moderator(ABC, Generic[StateT, ActionT]):
    @abstractmethod
    def moderate(self, x: Sequence[Union[StateT, ActionT]]) -> Sequence[float]:
        pass
