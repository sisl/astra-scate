"""
rewards.py

F: RewardComponent -> float
"""

from typing import Callable, List
from abc import ABC, abstractmethod

from astra_rl.core.state import ASTNode


class Reward(ABC):
    @abstractmethod
    def __call__(self, state: ASTNode) -> float:
        pass


class ASTReward(Reward):
    def __init__(
        self,
        moderator: Callable[[List[str]], List[float]],
        ppl: Callable[[List[str]], List[float]],
        weight_ppl: float = 0.1,
        weight_adv: float = 0.1,
        weight_combined: float = 0.1,
    ):
        self.__ppl = ppl
        self.__moderator = moderator

        self.weight_ppl = weight_ppl
        self.weight_adv = weight_adv
        self.weight_combined = weight_combined

    def ppl(self, x: List[str]) -> List[float]:
        return self.__ppl(x)

    def moderator(self, x: List[str]) -> List[float]:
        return self.__moderator(x)

    def __call__(self, state: ASTNode) -> float:
        combined_tox, defender_tox = self.moderator(
            [state.ast_ut + state.defender_ut, state.defender_ut]
        )
        (ppl_weight,) = self.ppl([state.ast_ut])

        reward = (
            self.weight_ppl * ppl_weight
            + self.weight_adv * defender_tox
            + self.weight_combined * combined_tox
        )

        return reward
