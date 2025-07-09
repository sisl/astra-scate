"""
state.py

Archoring data structure for the replay buffer.
i.e. each episode is one of these.
"""

from abc import ABC, abstractmethod
from typing import Sequence, Optional, Self, Any


class ASTNode(ABC):
    parent: Optional[Self]  # parent node (i.e. where this descends from)
    children: Sequence[Self]  # T(s'|s)
    reward: float  # R(s,a)

    ast_ut: str
    defender_ut: str


# TODO this seems like a bad name, can we find
# something more general (i.e., not graph dependent)
class ASTNodesFlattener(ABC):
    @abstractmethod
    def __call__(self, tree_root: ASTNode) -> Sequence[Any]:
        pass
