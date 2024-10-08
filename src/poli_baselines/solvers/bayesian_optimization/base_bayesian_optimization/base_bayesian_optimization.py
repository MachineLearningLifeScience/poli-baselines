from typing import Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    LogExpectedImprovement,
)
from botorch.fit import fit_gpytorch_mll_torch
from botorch.generation.gen import gen_candidates_torch
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.kernels import Kernel
from gpytorch.means import Mean
from gpytorch.mlls import ExactMarginalLogLikelihood
from poli.core.abstract_black_box import AbstractBlackBox

from poli_baselines.core.step_by_step_solver import StepByStepSolver
from poli_baselines.core.utils.visualization.bayesian_optimization import (
    plot_acquisition_in_2d,
    plot_prediction_in_2d,
)

from .bayesian_optimization_commons import (
    optimize_acquisition_function_using_grid_search,
)


class BaseBayesianOptimization(StepByStepSolver):
    """
    Base class for Bayesian Optimization solvers.

    Parameters
    ----------
    black_box : AbstractBlackBox
        The black box function to optimize.
    x0 : np.ndarray
        The initial input data.
    y0 : np.ndarray
        The initial output data.
    mean : Mean, optional
        The mean function of the Gaussian process model, by default None.
    kernel : Kernel, optional
        The kernel function of the Gaussian process model, by default None.
    acq_function : Type[AcquisitionFunction], optional
        The acquisition function to use, by default ExpectedImprovement.
    bounds : Tuple[float, float], optional
        The bounds of the input space, by default (-2.0, 2.0).
    penalize_nans_with : float, optional
        The value to assign to NaNs in the objective function, by default -10.0.

    Attributes
    ----------
    mean : Mean
        The mean function of the Gaussian process model.
    kernel : Kernel
        The kernel function of the Gaussian process model.
    acq_function : Type[AcquisitionFunction]
        The acquisition function to use.
    bounds : Tuple[float, float]
        The bounds of the input space.
    penalize_nans_with : float
        The value to assign to NaNs in the objective function.
    gp_model_of_objective : SingleTaskGP
        The Gaussian process model of the objective function.
        This starts being None, and is updated at every call
        to self._fit_model(...) (which is itself called at every
        call to self.next_candidate()).

    Methods
    -------
    _fit_model(model, x, y)
        Fits a Gaussian process model.
    _optimize_acquisition_function(acquisition_function)
        Optimizes an already instantiated acquisition function.
    _instantiate_acquisition_function(model)
        Instantiates the acquisition function.
    next_candidate()
        Runs one loop of Bayesian Optimization.
    plot_model_predictions(ax)
        Plots the model predictions in latent space (if the
        problem is 2D).

    Notes
    -----
    This class serves as a base class for implementing Bayesian Optimization solvers.
    It provides common functionality and methods that can be used by derived classes.
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
        Initialize the BaseBayesianOptimization class.

        Parameters
        ----------
        black_box : AbstractBlackBox
            The black box function to be optimized.
        x0 : np.ndarray
            The initial input points.
        y0 : np.ndarray
            The initial output values corresponding to the input points.
        mean : Mean, optional
            The mean function used in the Gaussian process, by default None.
        kernel : Kernel, optional
            The kernel function used in the Gaussian process, by default None.
        acq_function : Type[AcquisitionFunction], optional
            The acquisition function used for selecting the next point to evaluate, by default ExpectedImprovement.
        bounds : Tuple[float, float], optional
            The bounds of the input space, by default (-2.0, 2.0).
        penalize_nans_with : float, optional
            The value used to penalize NaN values in the acquisition function, by default -10.0.
        """
        super().__init__(black_box, x0, y0)
        self.mean = mean
        self.kernel = kernel
        self.acq_function = acq_function
        self.bounds = bounds
        self.penalize_nans_with = penalize_nans_with

        self.gp_model_of_objective = None

    def _fit_model(
        self,
        model: Type[SingleTaskGP],
        x: np.ndarray,
        y: np.ndarray,
    ) -> SingleTaskGP:
        """
        Fits a Gaussian process model.

        Parameters
        ----------
        model : Type[SingleTaskGP]
            The Gaussian process model to fit as a class, without being instantiated.
        x : np.ndarray
            The input data.
        y : np.ndarray
            The output data.

        Returns
        -------
        model : SingleTaskGP
            The fitted Gaussian process model.
        """
        x = torch.from_numpy(x).to(torch.float32)
        y = torch.from_numpy(y).to(torch.float32)
        model_instance = model(
            x,
            y,
            mean_module=self.mean,
            covar_module=self.kernel,
        )
        mll = ExactMarginalLogLikelihood(model_instance.likelihood, model_instance)
        fit_gpytorch_mll_torch(mll)
        model_instance.eval()

        self.gp_model_of_objective = model_instance

        return model_instance

    def _optimize_acquisition_function(
        self, acquisition_function: AcquisitionFunction
    ) -> np.ndarray:
        """
        Optimizes an already instantiated acquisition function.

        Parameters
        ----------
        acquisition_function : AcquisitionFunction
            The acquisition function to optimize.

        Returns
        -------
        candidate : np.ndarray
            The next candidate to evaluate.
        """
        n_dimensions = self.x0.shape[1]

        if n_dimensions in [1, 2]:
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
        """
        Instantiates the acquisition function.

        Parameters
        ----------
        model : SingleTaskGP
            The Gaussian process model.

        Returns
        -------
        acq_func : AcquisitionFunction
            The instantiated acquisition function.
        """
        _, y = self.get_history_as_arrays(penalize_nans_with=self.penalize_nans_with)
        if (
            self.acq_function == LogExpectedImprovement
            or self.acq_function == ExpectedImprovement
        ):
            acq_func = self.acq_function(model, best_f=y.max())
        else:
            raise NotImplementedError

        return acq_func

    def next_candidate(self) -> np.ndarray:
        """
        Runs one loop of Bayesian Optimization.

        Returns
        -------
        candidate : np.ndarray
            The next candidate to evaluate.

        Notes
        -----
        NaNs in the objective function are penalized by assigning them a value stored in self.penalize_nans_with.
        """
        raise NotImplementedError

    def plot_model_predictions(self, ax: plt.Axes) -> None:
        """
        Plots the model predictions in latent space.

        This method only works for problems with 2 dimensions, and
        can be used as a sanity check to see if the acquisition
        function is being computed correctly.

        Parameters:
        ----------
        ax: plt.Axes
            The matplotlib Axes object to plot the model predictions.
        """
        # TODO: assert that the x values are 2D.
        # TODO: add support for 1D spaces.

        # Plotting the model predictions in 2D.
        plot_prediction_in_2d(
            model=self.gp_model_of_objective,
            ax=ax,
            limits=self.bounds,
            historical_x=np.concatenate(self.history["x"], axis=0),
        )

    def plot_acquisition_function(self, ax: plt.Axes):
        """
        Plots the acquisition function in the input space.

        This method only works for problems with 2 dimensions, and
        can be used as a sanity check to see if the acquisition
        function is being computed correctly.

        Parameters:
        ----------
        ax: plt.Axes
            The matplotlib Axes object to plot the acquisition function.
        """
        # TODO: assert that the x values are 2D.
        # TODO: add support for 1D spaces.

        # Plotting the acquisition function in 2D.
        acq_func = self._instantiate_acquisition_function(
            model=self.gp_model_of_objective
        )
        plot_acquisition_in_2d(
            acq_function=acq_func,
            ax=ax,
            limits=self.bounds,
            historical_x=np.concatenate(self.history["x"], axis=0),
        )
