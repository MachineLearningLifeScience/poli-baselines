"""
This module implements a solver for vanilla Bayesian Optimization.

The BO backend will be BoTorch.

TODO: refactor the usual BO loop outside of these classes,
      using a generic implementation.
"""

from typing import Callable, Type, Tuple

import numpy as np

import torch


from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, AcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.generation.gen import gen_candidates_torch

from gpytorch.means import Mean
from gpytorch.kernels import Kernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from poli.core.abstract_black_box import AbstractBlackBox
from poli_baselines.core.abstract_solver import AbstractSolver

from .bayesian_optimization_commons import (
    optimize_acquisition_function_using_grid_search,
)


class VanillaBayesianOptimization(AbstractSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        mean: Mean = None,
        kernel: Kernel = None,
        acq_function: Type[AcquisitionFunction] = ExpectedImprovement,
        bounds: Tuple[float, float] = (-2.0, 2.0),
        penalize_nans_with: float = -10.0,
    ):
        """
        TODO: add docstring
        """
        super().__init__(black_box, x0, y0)
        self.mean = mean
        self.kernel = kernel
        self.acq_function = acq_function
        self.bounds = bounds
        self.penalize_nans_with = penalize_nans_with

    def next_candidate(self) -> np.ndarray:
        """Runs one loop of Bayesian Optimization.

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
        x = np.concatenate(self.history["x"], axis=0)
        y = np.concatenate(self.history["y"], axis=0)

        # Normalize the data
        # scaler_z = MinMaxScaler().fit(z)
        # scaler_y = MinMaxScaler().fit(y)
        # z = scaler_z.transform(z)
        # y = scaler_y.transform(y)

        # Penalize NaNs
        y[np.isnan(y)] = self.penalize_nans_with

        # Fit a GP
        model = SingleTaskGP(
            torch.from_numpy(x).to(torch.float32),
            torch.from_numpy(y).to(torch.float32),
            mean_module=self.mean,
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        model.eval()

        # Instantiate the acq. function
        if self.acq_function == ExpectedImprovement:
            acq_func = self.acq_function(model, best_f=y.max())
        else:
            raise NotImplementedError

        # Optimize the acquisition function
        # Bounds needs to be a 2xd tensor.
        # In this case, we can infer d from
        # the size of x.
        if x.shape[1] in [1, 2]:
            # We get the candidate by running a grid search
            # on the acquisition function.
            candidate = optimize_acquisition_function_using_grid_search(
                n_dimension=x.shape[1],
                acquisition_function=acq_func,
                bounds=self.bounds,
                n_points_in_acq_grid=100,
            )
        else:
            bounds_ = torch.tensor([list(self.bounds)] * x.shape[1]).T.to(torch.float32)
            candidate, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds_,
                q=1,
                num_restarts=20,
                raw_samples=100,
                gen_candidates=gen_candidates_torch,
            )
            candidate = candidate.numpy(force=True)

        # Unnormalize the candidate
        # candidate = scaler_z.inverse_transform(candidate)

        return candidate
