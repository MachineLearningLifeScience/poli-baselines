"""A solver that implements Line Bayesian Optimization [1].

Line Bayesian Optimization modifies the usual loop by only optimizing
the acquisition function along a single line in latent space. This line
can either be selected at random, be a coordinate direction, or use local
gradient information to choose a direction of likely descent. So far, we
only implement the random line and coordinate line search.

References
----------
[1] Kirschner, Johannes, Mojmir Mutny, Nicole Hiller, Rasmus Ischebeck, and Andreas Krause.
    “Adaptive and Safe Bayesian Optimization in High Dimensions via One-Dimensional Subspaces.”
    In Proceedings of the 36th International Conference on Machine Learning, 3429-38. PMLR, 2019.
    https://proceedings.mlr.press/v97/kirschner19a.html.
"""

from typing import Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction, LogExpectedImprovement
from botorch.models import SingleTaskGP
from gpytorch.kernels import Kernel
from gpytorch.means import Mean
from poli.core.abstract_black_box import AbstractBlackBox

from poli_baselines.core.utils.visualization.bayesian_optimization import \
    plot_prediction_in_2d
from poli_baselines.solvers.bayesian_optimization.base_bayesian_optimization.base_bayesian_optimization import \
    BaseBayesianOptimization

from .utilities import ray_box_intersection


class LineBO(BaseBayesianOptimization):
    """
    LineBO class represents a Bayesian Optimization solver that optimizes
    a black box function along a single line in the search space.

    Parameters
    ----------
    black_box : AbstractBlackBox
        The black box function to be optimized.
    x0 : np.ndarray
        The initial input points.
    y0 : np.ndarray
        The corresponding function values at the initial input points.
    mean : Mean, optional
        The mean function of the Gaussian process model, by default None.
    kernel : Kernel, optional
        The kernel function of the Gaussian process model, by default None.
    acq_function : type[AcquisitionFunction], optional
        The type of acquisition function to be used, by default ExpectedImprovement.
    bounds : Tuple[float, float], optional
        The bounds of the input space, by default (-2.0, 2.0).
    penalize_nans_with : float, optional
        The value to penalize NaN values in the acquisition function, by default -10.
    type_of_line : Literal["random", "coordinate"], optional
        The type of line to be used (random or coordinate), by default "random".

    Attributes
    ----------
    acq_function : type[AcquisitionFunction]
        The type of acquisition function to be used.
    bounds : Tuple[float, float]
        The bounds of the input space.
    type_of_line : Literal["random", "coordinate"]
        The type of line to be used (random or coordinate).
    gp_model_of_objective : None
        The GP model of the objective function.
    current_line : None
        The points in the current line.
    current_acq_values : None
        The values of the acquisition function in the current line.

    Methods
    -------
    _optimize_acquisition_function(acquisition_function)
        Optimizes the acquisition function along a single line in latent space.
    next_candidate()
        Encodes data to latent space, fits a Gaussian Process, and maximizes the acquisition function.
    plot_model_predictions(ax)
        Plots the model predictions in latent space.
    plot_acquisition_function_in_line(ax)
        Plots the acquisition function values along a line.

    References
    ----------
    [1] Kirschner, Johannes, Mojmir Mutny, Nicole Hiller, Rasmus Ischebeck, and Andreas Krause.
        “Adaptive and Safe Bayesian Optimization in High Dimensions via One-Dimensional Subspaces.”
        In Proceedings of the 36th International Conference on Machine Learning, 3429-38. PMLR, 2019.
        https://proceedings.mlr.press/v97/kirschner19a.html.
    """

    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        mean: Mean = None,
        kernel: Kernel = None,
        acq_function: type[AcquisitionFunction] = LogExpectedImprovement,
        bounds: Tuple[float, float] = (-2.0, 2.0),
        penalize_nans_with: float = -10,
        type_of_line: Literal["random", "coordinate"] = "random",
    ):
        """
        Initialize the LineBayesianOptimization object.

        Parameters
        ----------
        black_box : AbstractBlackBox
            The black box function to be optimized.
        x0 : np.ndarray
            The initial input points.
        y0 : np.ndarray
            The corresponding function values at the initial input points.
        mean : Mean, optional
            The mean function of the Gaussian process model, by default None.
        kernel : Kernel, optional
            The kernel function of the Gaussian process model, by default None.
        acq_function : type[AcquisitionFunction], optional
            The type of acquisition function to be used, by default ExpectedImprovement.
        bounds : Tuple[float, float], optional
            The bounds of the input space, by default (-2.0, 2.0).
        penalize_nans_with : float, optional
            The value to penalize NaN values in the acquisition function, by default -10.
        type_of_line : Literal["random", "coordinate"], optional
            The type of line to be used (random or coordinate), by default "random".
        """
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
        self.acq_function = acq_function
        self.bounds = bounds

        # The type of line (according to LineBO: either random or coordinate)
        # TODO: implement DescentLineBO.
        self.type_of_line = type_of_line

        # This variable contains the GP model of the objective,
        # it allows us to use the visualization utilities for Bayesian Optimization.
        self.gp_model_of_objective = None

        # This variable contains the points in the current line
        # and is mostly used for visualization purposes.
        self.current_line = None

        # This variable contains the values of the acquisition function
        # in the current line and is mostly used for visualization purposes.
        self.current_acq_values = None

    def _optimize_acquisition_function(
        self, acquisition_function: AcquisitionFunction
    ) -> np.ndarray:
        """
        Optimizes the acquisition function along a single line in latent space.

        The type of line is determined by the type_of_line attribute, which
        can be either "random" or "coordinate".

        Parameters
        ----------
        acquisition_function : AcquisitionFunction
            The acquisition function to optimize.

        Returns
        -------
        candidate : np.ndarray
            The candidate point.

        Notes
        -----
        - This class overwrites the method from BaseBayesianOptimization,
          optimizing the acqusition function across a single line in search
          space.
        """
        # The core difference of LineBO: optimize the acquisition function
        # over a random/coordinate linear direction in latent space.
        if self.type_of_line == "random":
            # Selecting a linear direction at random.
            line_ = np.random.randn(self.x0.shape[1])

            # Optimizing along this line
            # TODO: there must be a better way of
            # defining the line, especially since we're
            # interested in clipping the line to the bounds.
            best_x = self.get_best_solution()[0]

            # TODO: fix this:
            best_x = best_x.clip(*self.bounds)

            _, one_intersection = ray_box_intersection(
                best_x, line_, [self.bounds] * self.x0.shape[1]
            )
            _, another_intersection = ray_box_intersection(
                best_x, -line_, [self.bounds] * self.x0.shape[1]
            )
            t = np.linspace(0, 1, 100)
            xs_in_line = one_intersection[None, :] * t[:, None] + another_intersection[
                None, :
            ] * (1 - t[:, None])

        elif self.type_of_line == "coordinate":
            # Selecting a coordinate direction at random.
            line_ = np.zeros(self.x0.shape[1])
            line_[np.random.randint(self.x0.shape[1])] = 1.0

            # Optimizing along this line
            t = np.linspace(*self.bounds, 100)
            xs_in_line = t[:, None] * line_[None, :]
            # self.current_line = xs_in_line

        acq_values = acquisition_function(
            torch.from_numpy(xs_in_line).to(torch.float32).unsqueeze(1)
        )

        # More than one value might achieve the maximum,
        # so we select one at random
        candidates = xs_in_line[acq_values == acq_values.max()]
        candidate = candidates[np.random.randint(len(candidates))].reshape(1, -1)

        # Update the variables used for vizualization
        self.current_line = xs_in_line
        self.current_acq_values = acq_values

        return candidate

    def next_candidate(self) -> np.ndarray:
        """
        Encodes whatever data we have to latent space,
        fits a Gaussian Process, and maximizies the acquisition
        function.
        """
        # Build up the history
        x, y = self.get_history_as_arrays()

        # Penalize NaNs
        y[np.isnan(y)] = self.penalize_nans_with

        # Fit a GP
        model = self._fit_model(SingleTaskGP, x, y)

        # Update the model in the class itself
        self.gp_model_of_objective = model

        # Instantiate the acq. function
        acq_function = self._instantiate_acquisition_function(model)

        # Optimize the acquisition function
        candidate = self._optimize_acquisition_function(acq_function)

        return candidate

    def plot_model_predictions(self, ax: plt.Axes) -> None:
        """
        Plots the model predictions in latent space.

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

        # Plotting the current line
        ax.plot(
            self.current_line[:, 0],
            self.current_line[:, 1],
            c="black",
            linewidth=2,
            linestyle="--",
        )

    def plot_acquisition_function_in_line(self, ax: plt.Axes):
        """
        Plots the acquisition function values along a line.

        Parameters:
        ----------
        ax: plt.Axes
            The matplotlib Axes object to plot the acquisition function.
        """
        assert self.current_acq_values is not None
        ax.plot(
            np.linspace(-1, 1, len(self.current_acq_values.numpy(force=True))),
            self.current_acq_values.numpy(force=True),
        )
