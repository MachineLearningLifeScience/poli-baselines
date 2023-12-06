"""
This module implements SAASBO, a Bayesian optimization algorithm
that leverages certain priors on the kernel hyperparameters to
ensure the selection of sparse axis-aligned subspaces [1].

[1] High-Dimensional Bayesian Optimization with Sparse
    Axis-Aligned Subspaces, by Eriksson and Jankowiak, 2021.
    https://arxiv.org/abs/2103.00349
"""
from typing import Callable, Type, Tuple, Literal
from gpytorch.kernels import Kernel
from gpytorch.means import Mean

import numpy as np

from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, AcquisitionFunction
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

from poli.core.abstract_black_box import AbstractBlackBox
from poli_baselines.core.abstract_solver import AbstractSolver
from poli_baselines.solvers.bayesian_optimization.base_bayesian_optimization.base_bayesian_optimization import (
    BaseBayesianOptimization,
)


class SAASBO(BaseBayesianOptimization):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        mean: Mean = None,
        kernel: Kernel = None,
        acq_function: type[AcquisitionFunction] = ExpectedImprovement,
        bounds: Tuple[float, float] = (-2.0, 2.0),
        penalize_nans_with: float = -10,
    ):
        super().__init__(
            black_box=black_box,
            x0=x0,
            y0=y0,
            mean=mean,
            kernel=kernel,
            acq_function=acq_function,
            bounds=bounds,
            penalize_nans_with=penalize_nans_with,
        )

    def next_candidate(self) -> np.ndarray:
        """
        TODO: implement.
        """
        return super().next_candidate()
