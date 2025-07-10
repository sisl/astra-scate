"""
algorithm.py
"""

from abc import abstractmethod, ABC
from typing import Sequence, Generic

import torch

from astra_rl.core.problem import ASTRAProblem
from astra_rl.core.rollout import Graph
from astra_rl.core.common import Step, Batch, StateT, ActionT


class Algorithm(ABC, Generic[StateT, ActionT, Step, Batch]):
    def __init__(self, problem: ASTRAProblem[StateT, ActionT]):
        self.problem = problem

    @abstractmethod
    def flatten(self, graph: Graph[StateT, ActionT]) -> Sequence[Step]:
        pass

    @abstractmethod
    @staticmethod
    def collate_fn(batch: Sequence[Step]) -> Batch:
        """the collate_fn for torch dataloaders for batching"""
        pass

    @abstractmethod
    def step(self, batch: Batch) -> torch.Tensor:
        """take a step and compute loss"""
        pass
