"""
A simple solver that takes an encoder function,
encodes all the data points to latent space,
and runs Bayesian Optimization in latent space.

We use BoTorch as the backend for Bayesian Optimization.
"""

from typing import Callable, Type

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


class LatentSpaceBayesianOptimization(AbstractSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        encoder: Callable[[np.ndarray], np.ndarray],
        decoder: Callable[[np.ndarray], np.ndarray],
        acq_function: Type[AcquisitionFunction] = ExpectedImprovement,
    ):
        """
        TODO: add docstring

        encoder is a callable that takes [ints of classes] (np.array) -> [latent codes] (np.array)
        """
        super().__init__(black_box, x0, y0)
        self.encoder = encoder
        self.decoder = decoder
        self.acq_function = acq_function

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

        # Optimize the acquisition function
        # Bounds needs to be a 2xd tensor.
        # In this case, we can infer d from
        # the size of z.
        bounds = torch.tensor([[-2.0, 2.0]] * z.shape[1]).T

        # TODO: remove this grid search:
        if z.shape[1] == 2:
            n_points_in_acq_grid = 100
            limits = bounds[0].numpy(force=True)
            zs = torch.Tensor(
                [
                    [x, y]
                    for x in torch.linspace(*limits, n_points_in_acq_grid)
                    for y in reversed(torch.linspace(*limits, n_points_in_acq_grid))
                ]
            )
            acq_values = acq_func(zs.unsqueeze(1))
            candidate = zs[acq_values.argmax()]
        else:
            candidate, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=1,
                num_restarts=20,
                raw_samples=100,
                gen_candidates=gen_candidates_torch,
            )

        # Unnormalize the candidate
        candidate = candidate.detach().cpu().numpy().reshape(1, -1)
        # candidate = scaler_z.inverse_transform(candidate)
        print(candidate)

        # Return the next candidate
        candidate_as_ints = self.decoder(candidate)

        return candidate_as_ints
