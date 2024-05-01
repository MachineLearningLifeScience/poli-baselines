"""
This module provides the VanillaBayesianOptimization class, which performs Bayesian Optimization using a SingleTaskGP model.
"""

from typing import Type, Tuple

import torch
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from botorch.generation.gen import gen_candidates_torch

from botorch.models import SingleTaskGP
from botorch.acquisition import (
    ExpectedImprovement,
    AcquisitionFunction,
    LogExpectedImprovement,
)
from botorch.fit import fit_gpytorch_model

from gpytorch.means import Mean
from gpytorch.kernels import Kernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from poli.core.abstract_black_box import AbstractBlackBox
from poli_baselines.solvers.bayesian_optimization.base_bayesian_optimization.base_bayesian_optimization import (
    BaseBayesianOptimization,
)


class VanillaBayesianOptimization(BaseBayesianOptimization):
    """
    VanillaBayesianOptimization is a class that performs Bayesian Optimization using a SingleTaskGP model.

    Parameters
    ----------
    black_box : AbstractBlackBox
        The black box function to optimize.
    x0 : np.ndarray
        The initial input points.
    y0 : np.ndarray
        The initial output values.
    mean : Mean, optional
        The mean function of the Gaussian process model. If None, a default mean function is used.
    kernel : Kernel, optional
        The kernel function of the Gaussian process model. If None, a default kernel function is used.
    acq_function : Type[AcquisitionFunction], optional
        The acquisition function to optimize. If None, ExpectedImprovement is used.
    bounds : Tuple[float, float], optional
        The bounds of the input space. Default is (-2.0, 2.0).
    penalize_nans_with : float, optional
        The value to assign to NaNs in the objective function. Default is -10.0.

    Attributes
    ----------
    mean : Mean
        The mean function of the Gaussian process model.
    kernel : Kernel
        The kernel function of the Gaussian process model.
    acq_function : Type[AcquisitionFunction]
        The acquisition function to optimize.
    bounds : Tuple[float, float]
        The bounds of the input space.
    penalize_nans_with : float
        The value to assign to NaNs in the objective function.

    Methods
    -------
    next_candidate() -> np.ndarray
        Runs one loop of Bayesian Optimization using a SingleTaskGP model and returns the next candidate to evaluate.

    Notes
    -----
    - The NaNs in the objective function are penalized by assigning them a value stored in self.penalize_nans_with.
    - The black box function should implement the AbstractBlackBox interface.
    - The mean and kernels are passed to the SingleTaskGP.
    - The acquisition function should be a subclass of AcquisitionFunction.
    """

    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        mean: Mean = None,
        kernel: Kernel = None,
        acq_function: Type[AcquisitionFunction] = LogExpectedImprovement,
        bounds: Tuple[float, float] = (-2.0, 2.0),
        penalize_nans_with: float = -10.0,
    ):
        """
        Initialize the VanillaBayesianOptimization class.

        Parameters:
        ----------
        black_box : AbstractBlackBox
            The black box function to be optimized.
        x0 : np.ndarray
            The initial input samples.
        y0 : np.ndarray
            The initial output values.
        mean : Mean, optional
            The mean function for the Gaussian process, by default None.
        kernel : Kernel, optional
            The kernel function for the Gaussian process, by default None.
        acq_function : Type[AcquisitionFunction], optional
            The acquisition function to be used, by default ExpectedImprovement.
        bounds : Tuple[float, float], optional
            The bounds for the input samples, by default (-2.0, 2.0).
        penalize_nans_with : float, optional
            The value to penalize NaN values with, by default -10.0.
        """
        super().__init__(black_box, x0, y0)
        self.mean = mean
        self.kernel = kernel
        self.acq_function = acq_function
        self.bounds = bounds
        self.penalize_nans_with = penalize_nans_with

    def _fit_model(
        self, model: SingleTaskGP, x: np.ndarray, y: np.ndarray
    ) -> SingleTaskGP:
        x = torch.from_numpy(x).to(torch.float32)
        y = torch.from_numpy(y).to(torch.float32)

        model_instance = model(
            x,
            y,
            mean_module=self.mean,
            covar_module=self.kernel,
        )
        mll = ExactMarginalLogLikelihood(model_instance.likelihood, model_instance)
        fit_gpytorch_model(mll)
        model_instance.eval()

        self.gp_model_of_objective = model_instance

        return model_instance

    def next_candidate(self) -> np.ndarray:
        """Runs one loop of Bayesian Optimization using
        a SingleTaskGP.

        Returns
        -------
        candidate : np.ndarray
            The next candidate to evaluate.

        Notes
        -----
        We penalize the NaNs in the objective function by
        assigning them a value stored in self.penalize_nans_with.
        """
        # Build up the history
        x, y = self.get_history_as_arrays()

        # Penalize NaNs
        y[np.isnan(y)] = self.penalize_nans_with

        # Fit a GP
        model = self._fit_model(SingleTaskGP, x, y)

        # Instantiate the acquisition function
        acq_function = self._instantiate_acquisition_function(model)

        # Optimize the acquisition function
        candidate = self._optimize_acquisition_function(acq_function)
        # candidate = x_scaler.inverse_transform(candidate)

        return candidate
