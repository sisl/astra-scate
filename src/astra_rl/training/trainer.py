"""
trainer.py
This is a high-level trainer that imports the utilities in
harness.py to create a pushbutton interface for training.

Of course it will make a bunch of decisions for you. If
you don't want that, the use the interface in harness.py.
"""

import torch
from typing import Generic
from pydantic import BaseModel
from torch.optim import Optimizer

from astra_rl.training.harness import Harness
from astra_rl.core.environment import Environment
from astra_rl.core.algorithm import Algorithm
from astra_rl.core.common import ActionT, StateT, Batch, Step


class TrainingConfiguration(BaseModel):
    # optimization configuration
    lr: float = 3e-3
    batch_size: int = 16
    optimizer: str = "adamw"
    gradient_accumulation_steps: int = 1  # how many

    # training configuration
    training_steps: int = 1024  # how many rollouts to train for

    # rollout configuration
    num_episodes_per_experience: int = 8  # how many rollouts per gradient update


class Trainer(Generic[StateT, ActionT, Step, Batch]):
    optimizer: Optimizer

    def __init__(
        self,
        config: TrainingConfiguration,
        environment: Environment[StateT, ActionT],
        algorithm: Algorithm[StateT, ActionT, Step, Batch],
    ):
        self.config = config
        self.harness = Harness(
            environment, algorithm, config.num_episodes_per_experience
        )

        # TODO initialize LR scheduler?
        # ?????????????????????????????

        # initialize optimizer
        if config.optimizer == "adam":
            from torch.optim import Adam

            self.optimizer = Adam(environment.problem.parameters(), config.lr)
        elif config.optimizer == "adamw":
            from torch.optim import AdamW

            self.optimizer = AdamW(environment.problem.parameters(), config.lr)
        elif config.optimizer == "sgd":
            from torch.optim import SGD

            self.optimizer = SGD(environment.problem.parameters(), config.lr)
        elif config.optimizer == "rmsprop":
            from torch.optim import RMSprop

            self.optimizer = RMSprop(environment.problem.parameters(), config.lr)
        elif config.optimizer == "adagrad":
            from torch.optim import Adagrad

            self.optimizer = Adagrad(environment.problem.parameters(), config.lr)
        else:
            raise ValueError(f"Unknown optimizer configured: {config.optimizer}")

        # step counter, for acccmulutaion, etc.
        self._global_step_counter = 0

    def train(self) -> None:
        for _ in range(self.config.training_steps):
            buf = self.harness.experience()
            for batch in buf:
                # increment counter first for occumulation
                self._global_step_counter += 1
                loss: torch.Tensor = (
                    self.harness.step(batch) / self.config.gradient_accumulation_steps
                )
                # typing disabled here b/c mypy can't statically verify
                # that the loss has gradients
                loss.backward()  # type: ignore[no-untyped-call]

                # if gradient accumulation happens, step!
                if (
                    self._global_step_counter % self.config.gradient_accumulation_steps
                    == 0
                ):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
