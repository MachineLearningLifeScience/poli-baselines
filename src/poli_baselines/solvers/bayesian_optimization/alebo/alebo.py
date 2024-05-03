"""
Implementing ALEBO using Ax

ALEBO [1] stands for ...

[1] ...
"""

from __future__ import annotations

import numpy as np
from poli.core.abstract_black_box import AbstractBlackBox  # type: ignore
from poli.objective_repository import ToyContinuousBlackBox  # type: ignore

from poli_baselines.core.utils.ax.ax_solver import AxSolver


class ALEBO(AxSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        bounds: list[tuple[float, float]] | None = None,
        lower_dim: int = 4,
        noise_std: float = 0.0,
    ):
        self.noise_std = noise_std
        if bounds is None:
            assert isinstance(black_box, ToyContinuousBlackBox)
            bounds = [black_box.function.limits] * x0.shape[1]
        else:
            # If bounds is (lb, up), then we build the bounds
            # for the user
            if len(bounds) == 2:
                assert isinstance(bounds[0], (int, float))
                assert isinstance(bounds[1], (int, float))
                bounds = [bounds] * x0.shape[1]

            assert len(bounds) == x0.shape[1]
            assert all(len(bound) == 2 for bound in bounds)

        from ax.modelbridge.strategies.alebo import ALEBOStrategy  # type: ignore

        # Initialization is already being handled by this constructor,
        # so we leave init_size=1 (ideally it would be 0, but it is not
        # allowed by Ax's ALEBOStrategy)
        alebo_strategy = ALEBOStrategy(D=x0.shape[1], d=lower_dim, init_size=1)

        super().__init__(
            black_box=black_box,
            x0=x0,
            y0=y0,
            generation_strategy=alebo_strategy,
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
    x0 = np.random.uniform(-1.0, 1.0, (10, 50))
    y0 = black_box(x0)

    alebo_solver = ALEBO(
        black_box=black_box,
        x0=x0,
        y0=y0,
        lower_dim=2,
    )
    alebo_solver.solve(max_iter=50, verbose=True)
