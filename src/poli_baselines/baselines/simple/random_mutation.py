"""
This module implements a simple baseline for
black box optimization: performing mutations
at random.

Since the input is discrete the mutations
are performed by randomly selecting a
position in the input and randomly selecting
a new value for that position.
"""
from pathlib import Path

import numpy as np

from poli_baselines.core.abstract_solver import AbstractSolver
from poli.core.abstract_black_box import AbstractBlackBox


class RandomMutation(AbstractSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
    ):
        super().__init__(black_box, x0, y0)

    def next_candidate(self) -> np.ndarray:
        """
        Returns the next candidate solution
        after checking the history.

        In this case, the RandomMutation solver
        simply returns a random mutation of the
        best performing solution so far.
        """
        # Get the best performing solution so far
        best_x = self.history["x"][np.argmax(self.history["y"])]

        # Perform a random mutation
        x = best_x.copy()
        pos = np.random.randint(0, len(x))
        x[pos] = np.random.randint(0, self.black_box.L)

        return x
