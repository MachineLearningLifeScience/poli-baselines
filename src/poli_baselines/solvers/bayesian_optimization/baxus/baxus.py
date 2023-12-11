"""This module implements 'Bayesian Optimization with adaptively expanding subspaces' (BAXUS) [1].

This implementation is based on the tutorial provided inside BoTorch [2].

References
----------
[1] Increasing the scope as you learn: adaptive Bayesian
    Optimization in Nested Subspaces (TODO: complete).
[2] https://botorch.org/tutorials/baxus
"""

from typing import Tuple, Type
from botorch.models import SingleTaskGP
import numpy as np
from botorch.acquisition import AcquisitionFunction
from gpytorch.kernels import Kernel
from gpytorch.means import Mean

from poli.core.abstract_black_box import AbstractBlackBox

from poli_baselines.core.utils.acquisition.thompson_sampling import ThompsonSampling
from ..base_bayesian_optimization import BaseBayesianOptimization


class BAxUS(BaseBayesianOptimization):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        mean: Mean = None,
        kernel: Kernel = None,
        acq_function: Type[AcquisitionFunction] = ThompsonSampling,
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

    def _compute_trust_region(self) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def _optimize_acquisition_function(
        self, acquisition_function: AcquisitionFunction
    ) -> np.ndarray:
        ...

    def _instantiate_acquisition_function(
        self, model: SingleTaskGP
    ) -> AcquisitionFunction:
        ...

    def _compute_random_embedding_matrix(self) -> np.ndarray:
        ...

    def _expand_embedding_matrix(self) -> np.ndarray:
        ...

    def next_candidate(self) -> np.ndarray:
        ...
