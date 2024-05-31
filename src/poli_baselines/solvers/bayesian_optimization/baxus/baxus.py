from __future__ import annotations

from typing import List
import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox

from poli_baselines.core.abstract_solver import AbstractSolver

from baxus.benchmarks.benchmark_function import Benchmark
from baxus.baxus import BAxUS as OriginalBAxUS


class BAxUS(AbstractSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        upper_bound: float | List[float] | np.ndarray,
        lower_bound: float | List[float] | np.ndarray,
        noise_std: float,
        max_iter: int = 100,
        **kwargs,
    ):
        # super().__init__(black_box, x0, y0)
        self.black_box = black_box
        self.x0 = x0
        self.y0 = y0
        self.max_iter = max_iter
        _, n_dimensions = x0.shape

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
                print(y)
                return -y

        self._solver = OriginalBAxUS(
            Function(n_dimensions), target_dim=2, n_init=10, max_evals=self.max_iter
        )

    def solve(self, max_iter: int = 100, verbose: bool = False) -> None:
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
