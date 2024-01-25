"""
Implements a genetic algorithm solver using pymoo as a backend.
"""

from typing import Callable, Iterable
from typing_extensions import Self

import numpy as np
from numpy import ndarray
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

from poli.core.abstract_black_box import AbstractBlackBox
from poli_baselines.core.utils.pymoo.interface import (
    DiscretePymooProblem,
    _from_dict_to_array,
)

from poli_baselines.core.abstract_solver import AbstractSolver


class SaveHistoryCallback(Callback):
    def __init__(self, solver: AbstractSolver):
        super().__init__()
        self.solver = solver

    def notify(self, algorithm):
        # Since we're dealing with MixedVariables, we need to convert the
        # population of dicts {"x_i": value} to an array of shape [n, sequence_length].
        x_as_array = np.vstack(
            [_from_dict_to_array(x_i) for x_i in algorithm.pop.get("X")]
        )

        self.solver.update(x_as_array, -algorithm.pop.get("F"))


class GeneticAlgorithm(AbstractSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: ndarray,
        y0: ndarray,
        pop_size: int = 100,
        initialize_with_x0: bool = True,
    ):
        super().__init__(black_box, x0, y0)

        self.pymoo_problem = DiscretePymooProblem(
            black_box=self.black_box,
            x0=self.x0,
            y0=self.y0,
            initialize_with_x0=initialize_with_x0,
        )

        self.optimizer = MixedVariableGA(pop_size=pop_size)

    def next_candidate(self) -> ndarray:
        raise NotImplementedError

    def solve(
        self,
        max_iter: int = 100,
        break_at_performance: float = None,
        verbose: bool = False,
        pre_step_callbacks: Iterable[Callable[[Self], None]] = None,
        post_step_callbacks: Iterable[Callable[[Self], None]] = None,
    ) -> ndarray:
        res = minimize(
            self.pymoo_problem,
            self.optimizer,
            ("n_gen", max_iter),
            verbose=verbose,
            callback=SaveHistoryCallback(self),
        )

        return _from_dict_to_array([res.X]), -res.F


if __name__ == "__main__":
    from poli.objective_repository import AlohaProblemFactory

    f, x0, y0 = AlohaProblemFactory().create()

    solver = GeneticAlgorithm(black_box=f, x0=x0, y0=y0, pop_size=10)

    x = solver.solve(max_iter=100, verbose=True)
    print(x)
