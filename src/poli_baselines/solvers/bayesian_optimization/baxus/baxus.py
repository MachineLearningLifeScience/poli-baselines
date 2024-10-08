from __future__ import annotations

import warnings
from typing import List

import numpy as np
from baxus.baxus import BAxUS as OriginalBAxUS
from baxus.benchmarks.benchmark_function import Benchmark
from poli.core.abstract_black_box import AbstractBlackBox

from poli_baselines.core.abstract_solver import AbstractSolver


class BAxUS(AbstractSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        bounds: tuple[float, float],
        noise_std: float,
        max_iter: int = 100,
        target_dim: int = 2,
        n_dimensions: int = None,
        n_init: int = 10,
        **kwargs,
    ):
        # super().__init__(black_box, x0, y0)
        self.black_box = black_box
        self.x0 = x0
        self.y0 = y0
        self.max_iter = max_iter
        self.bounds = bounds
        self.noise_std = noise_std
        self.target_dim = target_dim
        self.n_init = n_init
        lower_bound, upper_bound = bounds

        if n_dimensions is None:
            if x0 is None:
                raise ValueError(
                    "You should provide either x0 or n_dimensions in the constructor."
                )
            _, n_dimensions = x0.shape

        if x0 is not None or y0 is not None:
            warnings.warn(
                "Since we follow the original implementation, "
                "BAxUS does not use x0 or y0 for initialization. "
                "By default, it uses n_init points in a SOBOL sequence. "
                "Specify how many points you would like to initialize with "
                "by using the n_init parameter."
            )

        # Making sure that lower and upper bounds have the same shape as x0
        if isinstance(upper_bound, float):
            upper_bound = np.full(n_dimensions, upper_bound)
        if isinstance(lower_bound, float):
            lower_bound = np.full(n_dimensions, lower_bound)

        class Function(Benchmark):
            def __init__(
                self,
                dim: int,
            ):
                super().__init__(dim, upper_bound, lower_bound, noise_std)

            def __call__(self, x: np.ndarray | List[float] | List[List[float]]):
                # Making sure that x has the right shape
                assert isinstance(x, np.ndarray)
                x = x.reshape(1, n_dimensions)
                y = black_box(x).flatten()[0]
                return -y

        self.baxus_benchmark = Function(n_dimensions)

    def solve(self, max_iter: int = 100, verbose: bool = False) -> None:
        if max_iter is None:
            max_iter = self.max_iter
            assert self.max_iter is not None, (
                "You should provide max_iter, be it in "
                "the constructor or in the solve method."
            )
        self._solver = OriginalBAxUS(
            self.baxus_benchmark,
            target_dim=self.target_dim,
            n_init=self.n_init,
            max_evals=max_iter + self.n_init,
        )
        self._solver.optimize()


if __name__ == "__main__":
    from poli_baselines.core.utils.isolation.registry import register_solver

    register_solver(
        solver_class=BAxUS,
        conda_environment_name="baxus_",
    )

    # from poli.objective_repository import ToyContinuousBlackBox

    # black_box = ToyContinuousBlackBox(
    #     function_name="branin_2d",
    #     n_dimensions=2,
    #     embed_in=50,
    #     dimensions_to_embed_in=[19, 34],
    # )
    # x0 = np.random.uniform(-1.0, 1.0, (10, 50))
    # y0 = black_box(x0)

    # baxus_solver = BAxUS_(
    #     black_box=black_box,
    #     x0=x0,
    #     y0=y0,
    #     upper_bound=black_box.bounds[1],
    #     lower_bound=black_box.bounds[0],
    #     noise_std=0.01,
    # )
    # baxus_solver.solve(max_iter=150, verbose=True)


if __name__ == "__main__":
    ...
