from typing import Iterable, Callable, Tuple

import numpy as np
from poli.core.abstract_black_box import AbstractBlackBox

from poli_baselines.core.abstract_solver import AbstractSolver
from poli_baselines.solvers.bayesian_optimization.turbo.poli_function_wrapper import PoliFunctionWrapper
from poli_baselines.solvers.bayesian_optimization.turbo.turbo import TurboM


class TurboWrapper(AbstractSolver):
    def __init__(self, black_box: AbstractBlackBox, x0: np.ndarray, y0: np.ndarray):
        super().__init__(black_box, x0, y0)

    def solve(
        self,
        max_iter: int = 100,
        break_at_performance: float = None,
        verbose: bool = False,
        pre_step_callbacks=None,
        post_step_callbacks=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: get actual bounds!
        ub = np.ones(self.x0.shape[1])
        lb = -ub
        # TODO: let turbo know of initial observations
        turbo_m = TurboM(
            f=PoliFunctionWrapper(self.black_box),  # Handle to objective function
            lb=lb,  # Numpy array specifying lower bounds
            ub=ub,  # Numpy array specifying upper bounds
            n_init=10,  # Number of initial bounds from an Symmetric Latin hypercube design
            max_evals=1000,  # Maximum number of evaluations
            n_trust_regions=5,  # Number of trust regions
            batch_size=10,  # How large batch size TuRBO uses
            verbose=True,  # Print information from each batch
            use_ard=True,  # Set to true if you want to use ARD for the GP kernel
            max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
            n_training_steps=50,  # Number of steps of ADAM to learn the hypers
            min_cuda=1024,  # Run on the CPU for small datasets
            device="cpu",  # "cpu" or "cuda"
            dtype="float64",  # float64 or float32
        )
        turbo_m.optimize()
        idx = np.argmin(turbo_m.fX)
        return turbo_m.X[idx, ...], turbo_m.fX[idx]
