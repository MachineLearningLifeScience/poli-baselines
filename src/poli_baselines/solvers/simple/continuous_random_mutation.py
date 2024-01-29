"""This module implements continuous random search by mutating
the best performing candidate, adding Gaussian noise."""

from typing import Tuple
import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli_baselines.core.abstract_solver import AbstractSolver


class ContinuousRandomMutation(AbstractSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        loc: float = 0.0,
        std: float = 1.0,
        bounds: Tuple[float, float] = (-2.0, 2.0),
        penalize_nans_with: float = -10.0,
    ):
        """
        TODO: add docstring
        """
        super().__init__(black_box, x0, y0)
        self.loc = loc
        self.std = std
        self.bounds = bounds
        self.penalize_nans_with = penalize_nans_with

    def next_candidate(self) -> np.ndarray:
        """Takes the best candidate, and adds Gaussian noise.

        Returns
        -------
        candidate : np.ndarray
            The next candidate to evaluate.

        Notes
        -----
        We ignore NaNs in the objective function
        """
        # Get the best performing solution so far
        best_x = self.history["x"][np.nanargmax(self.history["y"])]

        # Perform a random mutation
        candidate = best_x + np.random.normal(self.loc, self.std, size=best_x.shape)

        return candidate
