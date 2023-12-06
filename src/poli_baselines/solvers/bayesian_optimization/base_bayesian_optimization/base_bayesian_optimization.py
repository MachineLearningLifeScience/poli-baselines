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


class BaseBayesianOptimization(AbstractSolver):
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

    def _fit_model(
        self, model: Type[SingleTaskGP], x: np.ndarray, y: np.ndarray
    ) -> SingleTaskGP:
        """Fits a GP model.

        Parameters
        ----------
        model : Type[SingleTaskGP]
            The GP model to fit as a class, without being instantiated.
        x : np.ndarray
            The input data.
        y : np.ndarray
            The output data.

        Returns
        -------
        model : SingleTaskGP
            The fitted GP model.
        """
        model_instance = model(
            torch.from_numpy(x).to(torch.float32),
            torch.from_numpy(y).to(torch.float32),
            mean_module=self.mean,
        )
        mll = ExactMarginalLogLikelihood(model_instance.likelihood, model_instance)
        fit_gpytorch_model(mll)
        model_instance.eval()

        return model_instance

    def _optimize_acquisition_function(
        self, acquisition_function: AcquisitionFunction
    ) -> np.ndarray:
        """Optimizes an already instantiated acquisition function.

        Parameters
        ----------
        acquisition_function : AcquisitionFunction
            The acquisition function to optimize.

        Returns
        -------
        candidate : np.ndarray
            The next candidate to evaluate.
        """
        # Compute the number of dimensions of the input.
        # In this case, we can infer d from
        # the size of x.
        n_dimensions = self.x0.shape[1]

        # Optimize the acquisition function
        # Bounds needs to be a 2xd tensor.
        if n_dimensions in [1, 2]:
            # We get the candidate by running a grid search
            # on the acquisition function.
            # TODO: implement this for d = 1.
            candidate = optimize_acquisition_function_using_grid_search(
                n_dimension=n_dimensions,
                acquisition_function=acquisition_function,
                bounds=self.bounds,
                n_points_in_acq_grid=100,
            )
        else:
            bounds_ = torch.tensor([list(self.bounds)] * n_dimensions).T.to(
                torch.float32
            )
            candidate, _ = optimize_acqf(
                acq_function=acquisition_function,
                bounds=bounds_,
                q=1,
                num_restarts=20,
                raw_samples=100,
                gen_candidates=gen_candidates_torch,
            )
            candidate = candidate.numpy(force=True)

        return candidate

    def _instantiate_acquisition_function(
        self, model: SingleTaskGP
    ) -> AcquisitionFunction:
        """Instantiates the acquisition function.

        Parameters
        ----------
        model : SingleTaskGP
            The GP model.

        Returns
        -------
        acq_func : AcquisitionFunction
            The instantiated acquisition function.
        """
        if self.acq_function == ExpectedImprovement:
            acq_func = self.acq_function(model, best_f=self.y0.max())
        else:
            raise NotImplementedError

        return acq_func

    def next_candidate(self) -> np.ndarray:
        """An abstract method that runs one loop of Bayesian Optimization.

        Returns
        -------
        candidate : np.ndarray
            The next candidate to evaluate.

        Notes
        -----
        We penalize the NaNs in the objective function by
        assigning them a value stored in self.penalize_nans_with.
        """
        raise NotImplementedError
