"""
algorithm.py
"""

from abc import abstractmethod, ABC
from typing import Any, Dict, List

import torch

from astra_rl.core.state import ASTNodesFlattener


class Algorithm(ABC):
    @abstractmethod
    @property
    def flattener(self) -> ASTNodesFlattener:
        pass

    @abstractmethod
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Any:
        """the collate_fn for torch dataloaders for batching"""
        pass

    @abstractmethod
    def step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """take a step and comput loss"""
        pass
