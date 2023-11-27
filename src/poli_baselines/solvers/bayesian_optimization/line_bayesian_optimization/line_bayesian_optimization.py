"""
A simple solver that takes an encoder function,
encodes all the data points to latent space,
and runs Bayesian Optimization in latent space.

We use BoTorch as the backend for Bayesian Optimization.
"""

from typing import Callable, Type, Tuple, Literal

import matplotlib.pyplot as plt
import numpy as np

import torch


from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, AcquisitionFunction

from gpytorch.mlls import ExactMarginalLogLikelihood

from poli.core.abstract_black_box import AbstractBlackBox
from poli_baselines.core.abstract_solver import AbstractSolver
from poli_baselines.core.utils.visualization.bayesian_optimization import (
    plot_prediction_in_2d,
    plot_acquisition_in_2d,
)

from .utilities import ray_box_intersection


class LineBO(AbstractSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        acq_function: Type[AcquisitionFunction] = ExpectedImprovement,
        bounds: Tuple[float, float] = (-2.0, 2.0),
        type_of_line: Literal["random", "coordinate"] = "random",
    ):
        """
        TODO: add docstring

        encoder is a callable that takes [ints of classes] (np.array) -> [latent codes] (np.array)
        """
        super().__init__(black_box, x0, y0)
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

    def next_candidate(self) -> np.ndarray:
        """
        Encodes whatever data we have to latent space,
        fits a Gaussian Process, and maximizies the acquisition
        function.
        """
        # Encode the data to latent space
        x = np.concatenate(self.history["x"], axis=0)
        y = np.concatenate(self.history["y"], axis=0)

        # Normalize the data
        # scaler_z = MinMaxScaler().fit(z)
        # scaler_y = MinMaxScaler().fit(y)
        # z = scaler_z.transform(z)
        # y = scaler_y.transform(y)

        # Penalize NaNs (TODO: add a flag for this)
        y[np.isnan(y)] = -10.0

        # Fit a GP
        model = SingleTaskGP(
            torch.from_numpy(x).to(torch.float32),
            torch.from_numpy(y).to(torch.float32),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        model.eval()

        # Update the model in the class itself:
        self.gp_model_of_objective = model

        # Instantiate the acq. function
        if self.acq_function == ExpectedImprovement:
            acq_func = self.acq_function(model, best_f=y.max())
        else:
            raise NotImplementedError

        # The core difference of LineBO: optimize the acquisition function
        # over a random/coordinate linear direction in latent space.
        if self.type_of_line == "random":
            # Selecting a linear direction at random.
            l = np.random.randn(x.shape[1])

            # Optimizing along this line
            # TODO: there must be a better way of
            # defining the line, especially since we're
            # interested in clipping the line to the bounds.
            last_x = x[-1]
            _, one_intersection = ray_box_intersection(
                last_x, l, [self.bounds] * x.shape[1]
            )
            _, another_intersection = ray_box_intersection(
                last_x, -l, [self.bounds] * x.shape[1]
            )
            t = np.linspace(0, 1, 100)
            xs_in_line = one_intersection[None, :] * t[:, None] + another_intersection[
                None, :
            ] * (1 - t[:, None])

        elif self.type_of_line == "coordinate":
            # Selecting a coordinate direction at random.
            l = np.zeros(x.shape[1])
            l[np.random.randint(x.shape[1])] = 1.0

            # Optimizing along this line
            t = np.linspace(*self.bounds, 100)
            xs_in_line = t[:, None] * l[None, :]
            # self.current_line = xs_in_line

        acq_values = acq_func(
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

    def plot_model_predictions(self, ax: plt.Axes) -> None:
        """
        Plots the model predictions in latent space.
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
        TODO: implement.
        """
        assert self.current_acq_values is not None
        ax.plot(
            np.linspace(-1, 1, len(self.current_acq_values.numpy(force=True))),
            self.current_acq_values.numpy(force=True),
        )
