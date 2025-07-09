"""
problem.py
Generic class of an AstraProblem
"""

from abc import ABC, abstractmethod
from typing import List, Any, Dict
from collections import defaultdict

import torch

from astra_rl.core.moderator import Moderator
from astra_rl.logging import logger


class ASTRAProblem(ABC):
    """
    A problem is defined by
    - a dataset
    - models (defense, adversary, baseline) + how to call them
    - moderator
    - reward formulation
    """

    def __init__(self, moderator: Moderator):
        self.moderator: Moderator = moderator

        # we check all asserts once, and then disable them
        self.__disable_asserts: Dict[str, bool] = defaultdict(bool)

    @abstractmethod
    def get_target_logprobs(
        self, context: List[str], continuation: List[str]
    ) -> torch.Tensor:
        """Evaluates P(continuation|context) on *model under test*.

        Args:
            context (List[str]): List of strings, where each string is a context on which the
                                 continuation's probability is conditioned.
            continuation (List[str]): List of strings, where each string is a continuation whose
                                      probability is measured.

        Note:
            This should be batched; i.e., len(context) == len(continuation) and each
            represents a batch element.

        Returns:
            torch.Tensor: The log probabilities of the continuations given their contexts.
        """

        pass

    @abstractmethod
    def get_baseline_logprobs(
        self, context: List[str], continuation: List[str]
    ) -> torch.Tensor:
        """Evaluates P(continuation|context) on *attacker's baseline distribution* for KL
           divergence measurements.

        Args:
            context (List[str]): List of strings, where each string is a context on which the
                                 continuation's probability is conditioned.
            continuation (List[str]): List of strings, where each string is a continuation whose
                                      probability is measured.

        Note:
            This should be batched; i.e., len(context) == len(continuation) and each
            represents a batch element. Note that this is *not* the defender's model, but
            rather the baseline model used for measuring KL divergence to make sure that
            the trained attacker stays an LM.

        Returns:
            torch.Tensor: The log probabilities of the continuations given their contexts.
        """

        pass

    @abstractmethod
    def get_attacker_logprobs(
        self, context: List[str], continuation: List[str]
    ) -> torch.Tensor:
        """Evaluates P(continuation|context) on *attacker*. This must return tensor w/ grads!

        Args:
            context (List[str]): List of strings, where each string is a context on which the
                                 continuation's probability is conditioned.
            continuation (List[str]): List of strings, where each string is a continuation whose
                                      probability is measured.

        Note:
            This should be batched; i.e., len(context) == len(continuation) and each
            represents a batch element.

        Returns:
            torch.Tensor: The log probabilities of the continuations given their contexts.
        """

        pass

    @abstractmethod
    def rollout_prompt_with_attacker(self, x: List[str]) -> List[str]:
        """Rolls out the prompt with the attacker model. Do *not* return the prompt.

        Args:
            x (List[str]): List of strings representing the prompt to be rolled out.

        Returns:
            List[str]: The rolled out prompt with the adversary model.
        """
        pass

    @abstractmethod
    def rollout_prompt_with_target(self, x: List[str]) -> List[str]:
        """Rolls out the prompt with the model under test. Do *not* return the prompt.

        Args:
            x (List[str]): List of strings representing the prompt to be rolled out.

        Returns:
            List[str]: The rolled out prompt with the adversary model.
        """
        pass

    @abstractmethod
    @property
    def flattener(self) -> Any:
        pass

    @abstractmethod
    def reward(self, step: Any) -> float:
        pass

    def __check_continuation(
        self, check_key: str, context: List[str], continuation: List[str]
    ) -> None:
        if self.__disable_asserts[check_key]:
            return
        # make sure that we didn't repeat context in continuation
        if len(context) > 0 and len(continuation) > 0:
            assert context[0] not in continuation[0], (
                "Context should not be repeated in continuation."
            )
        self.__disable_asserts[check_key] = True

    def __check_logprobs(
        self,
        check_key: str,
        logprobs: torch.Tensor,
        ctx_length: int,
        requires_grad: bool = False,
    ) -> None:
        if self.__disable_asserts[check_key]:
            return
        # check that logprobs is a tensor and has gradients
        assert isinstance(logprobs, torch.Tensor), (
            "Attacker logprobs must be a torch.Tensor."
        )
        if requires_grad:
            assert logprobs.requries_grad, "Attacker logprobs must be a torch.Tensor."
        # check that the size of the tensor is B x 1, where B is the batch size
        if logprobs.dim() == 1:
            logprobs = logprobs.unsqueeze(1)
        assert logprobs.dim() == 2, (
            "Attacker logprobs must be a 2D tensor (B, 1) or a 1D list of numbers (B,)."
        )
        # check that the first dimension is the batch size
        assert logprobs.size(0) == ctx_length, (
            "Attacker logprobs must have the same batch size as the context."
        )
        # warn if everything is between 0 and 1
        if ((logprobs >= 0.0) & (logprobs <= 1.0)).all():
            logger.warning(
                "Attacker *log*probs looks suspiciously like probabilities, "
                "try taking the .log() of your tensor?"
            )
        self.__disable_asserts[check_key] = True

    def __get_attacker_logprobs_and_validate(
        self, context: List[str], continuation: List[str]
    ) -> torch.Tensor:
        logprobs = self.get_attacker_logprobs(context, continuation)
        self.__check_logprobs("attacker_logprobs", logprobs, len(context), True)
        return logprobs

    def __get_target_logprobs_and_validate(
        self, context: List[str], continuation: List[str]
    ) -> torch.Tensor:
        logprobs = self.get_target_logprobs(context, continuation)
        self.__check_logprobs("target_logprobs", logprobs, len(context), False)
        return logprobs

    def __get_baseline_logprobs_and_validate(
        self, context: List[str], continuation: List[str]
    ) -> torch.Tensor:
        logprobs = self.get_baseline_logprobs(context, continuation)
        self.__check_logprobs("baseline_logprobs", logprobs, len(context), False)
        return logprobs

    def __rollout_prompt_with_attacker_and_validate(self, x: List[str]) -> List[str]:
        rolled_out = self.rollout_prompt_with_attacker(x)
        self.__check_continuation("attacker_rollout", x, rolled_out)
        return rolled_out

    def __rollout_prompt_with_target_and_validate(self, x: List[str]) -> List[str]:
        rolled_out = self.rollout_prompt_with_target(x)
        self.__check_continuation("target_rollout", x, rolled_out)
        return rolled_out
