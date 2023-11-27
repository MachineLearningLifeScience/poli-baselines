"""
In this script, we use pymoo's implementation of NSGA-II to solve a multi-objective problem.
"""
from typing import List, Union, Callable

import numpy as np

from poli.core.multi_objective_black_box import MultiObjectiveBlackBox
from poli.core.abstract_black_box import AbstractBlackBox

from poli_baselines.core.abstract_solver import AbstractSolver
from poli_baselines.core.utils.pymoo.interface import DiscretePymooProblem


class NSGASolver(AbstractSolver):
    """
    This class implements a solver for multi-objective problems
    using NSGA-II. Since NSGA-II is a solver for continuous problems,
    we need to encode the inputs and decode the outputs.

    In other words, NSGA-II runs in latent space
    """

    def __init__(
        self,
        black_box: Union[MultiObjectiveBlackBox, List[AbstractBlackBox]],
        x0: np.ndarray,
        y0: np.ndarray,
        population_size: int = 100,
    ):
        if isinstance(black_box, list):
            black_box = MultiObjectiveBlackBox(
                L=x0.shape[1], objective_functions=black_box
            )
        super().__init__(black_box, x0, y0)

        self.population_size = population_size
        self.inner_pymoo_problem = DiscretePymooProblem(
            black_box=self.black_box,
            x0=self.x0,
            y0=self.y0,
        )

    def next_candidate(self) -> np.ndarray:
        return super().next_candidate()
