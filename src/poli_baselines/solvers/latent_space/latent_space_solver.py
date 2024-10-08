"""Implements an abstract latent space solver"""

from typing import Callable, Tuple, Type

import numpy as np
from poli.core.abstract_black_box import AbstractBlackBox

from poli_baselines.core.step_by_step_solver import AbstractSolver, StepByStepSolver


class LatentSpaceSolver(StepByStepSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        continuous_optimizer_class: Type[AbstractSolver],
        encoder: Callable[[np.ndarray], np.ndarray],
        decoder: Callable[[np.ndarray], np.ndarray],
        **kwargs_for_continuous_optimizer,
    ):
        super().__init__(black_box, x0, y0)

        continuous_optimizer = continuous_optimizer_class(
            black_box=black_box,
            x0=encoder(x0),
            y0=y0,
            **kwargs_for_continuous_optimizer,
        )
        self.continuous_optimizer = continuous_optimizer
        self.encoder = encoder
        self.decoder = decoder

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs the solver for one iteration.
        """
        z = self.continuous_optimizer.next_candidate()
        x = self.decoder(z)
        y = self.black_box(x)

        # TODO: in an ideal world, we would
        # only maintain a single history. We could
        # update history to be a property instead.

        # Updating this solver's history
        self.update(x, y)
        self.post_update(x, y)

        # Updating the continuous optimizer's history
        self.continuous_optimizer.update(z, y)
        self.continuous_optimizer.post_update(z, y)
        self.iteration += 1

        return x, y
