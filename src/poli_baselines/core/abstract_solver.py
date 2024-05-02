from typing import Union

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox


class AbstractSolver:
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: Union[np.ndarray, None] = None,
        y0: Union[np.ndarray, None] = None,
    ):
        self.black_box = black_box
        self.x0 = x0
        self.y0 = y0

    def solve(
        self,
        max_iter: int = 100,
        n_initial_points: int = 0,
        seed: int = None,
    ) -> None:
        """
        Optimizes the problem for a given number of iterations.

        Logging of the black box calls is usually handled by
        poli and their observers.

        Parameters
        ----------
        max_iter: int, optional
            The maximum number of iterations to run. By default, 100.
        n_initial_points: int, optional
            The number of initial points to evaluate before starting
            the optimization. By default, 0.
        seed: int, optional
            The seed to use for the random number generator. By default,
            None, which means that no seed is set.
        """
        raise NotImplementedError
