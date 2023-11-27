"""
A simple solver that takes an encoder function,
encodes all the data points to latent space,
and runs Bayesian Optimization in latent space.

We use BoTorch as the backend for Bayesian Optimization.
"""

from typing import Callable, Type, Tuple

import numpy as np

import torch


from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, AcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.generation.gen import gen_candidates_torch

from gpytorch.mlls import ExactMarginalLogLikelihood

from poli.core.abstract_black_box import AbstractBlackBox
from poli_baselines.core.abstract_solver import AbstractSolver


class LatentSpaceLineBO(AbstractSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        encoder: Callable[[np.ndarray], np.ndarray],
        decoder: Callable[[np.ndarray], np.ndarray],
        acq_function: Type[AcquisitionFunction] = ExpectedImprovement,
        bounds: Tuple[float, float] = (-2.0, 2.0),
        type_of_line: str = "random",
    ):
        """
        TODO: add docstring

        encoder is a callable that takes [ints of classes] (np.array) -> [latent codes] (np.array)
        """
        super().__init__(black_box, x0, y0)
        self.encoder = encoder
        self.decoder = decoder
        self.acq_function = acq_function
        self.bounds = bounds
        self.type_of_line = type_of_line

    def next_candidate(self) -> np.ndarray:
        """
        Encodes whatever data we have to latent space,
        fits a Gaussian Process, and maximizies the acquisition
        function.
        """
        # Encode the data to latent space
        x = np.concatenate(self.history["x"], axis=0)
        y = np.concatenate(self.history["y"], axis=0)

        z = self.encoder(x)

        # Normalize the data
        # scaler_z = MinMaxScaler().fit(z)
        # scaler_y = MinMaxScaler().fit(y)
        # z = scaler_z.transform(z)
        # y = scaler_y.transform(y)

        # Penalize NaNs (TODO: add a flag for this)
        y[np.isnan(y)] = -10.0

        # Fit a GP
        model = SingleTaskGP(
            torch.from_numpy(z).to(torch.float32),
            torch.from_numpy(y).to(torch.float32),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        model.eval()

        # Instantiate the acq. function
        if self.acq_function == ExpectedImprovement:
            acq_func = self.acq_function(model, best_f=y.max())
        else:
            raise NotImplementedError

        # The core difference of LineBO: optimize the acquisition function
        # over a random/coordinate linear direction in latent space.
        if self.type_of_line == "random":
            # Selecting a linear direction at random.
            offset = np.random.randn(z.shape[1])
            l = np.random.randn(z.shape[1])

            # Optimizing along this line
            t = np.linspace(-1.0, 1.0, 100)
            zs_in_line = offset + t[:, None] * l[None, :]
        elif self.type_of_line == "coordinate":
            # Selecting a coordinate direction at random.
            l = np.zeros(z.shape[1])
            l[np.random.randint(z.shape[1])] = 1.0

            # Optimizing along this line
            t = np.linspace(*self.bounds, 100)
            zs_in_line = t[:, None] * l[None, :]

        acq_values = acq_func(torch.from_numpy(zs_in_line).to(torch.float32))
        candidate = zs_in_line[acq_values == acq_values.max()]
        candidate = candidate.detach().cpu().numpy().reshape(1, -1)

        # Return the next candidate
        candidate_as_ints = self.decoder(candidate)

        return candidate_as_ints
