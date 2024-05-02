"""
Implementing SAASBO using Ax

SAASBO [1] stands for Sparse Axis-Aligned Subspace Bayesian Optimization,
and was proposed by Eriksson and ...

[1] ...
"""

import numpy as np
from poli.core.abstract_black_box import AbstractBlackBox  # type: ignore[import]
from poli.objective_repository import ToyContinuousBlackBox  # type: ignore[import]

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy  # type: ignore[import]
from ax.modelbridge.registry import Models  # type: ignore[import]

from hdbo_benchmark.utils.ax.ax_solver import AxSolver


class SAASBO(AxSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        bounds: list[tuple[float, float]] | None = None,
        noise_std: float = 0.0,
    ):
        self.noise_std = noise_std
        generation_strategy = GenerationStrategy(
            steps=[
                # GenerationStep(
                #     model=Models.SOBOL,
                #     num_trials=10,
                # ),
                GenerationStep(
                    model=Models.SAASBO,
                    num_trials=-1,
                    max_parallelism=black_box.num_workers,
                ),
            ]
        )
        super().__init__(
            black_box=black_box,
            x0=x0,
            y0=y0,
            generation_strategy=generation_strategy,
            bounds=bounds,
            noise_std=noise_std,
        )


if __name__ == "__main__":
    from poli.objective_repository import ToyContinuousBlackBox

    black_box = ToyContinuousBlackBox(
        function_name="branin_2d",
        n_dimensions=2,
        embed_in=50,
        dimensions_to_embed_in=[19, 34],
    )
    x0 = np.random.uniform(-1.0, 1.0, (1, 50))
    y0 = black_box(x0)

    saasbo_solver = SAASBO(
        black_box=black_box,
        x0=x0,
        y0=y0,
    )
    saasbo_solver.solve(max_iter=50, verbose=True)
