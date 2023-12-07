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

from gpytorch.mlls import ExactMarginalLogLikelihood

from poli.core.abstract_black_box import AbstractBlackBox
from poli_baselines.core.abstract_solver import AbstractSolver


# TODO: this solver should inherit from BaseBayesianOptimization instead.
class LatentSpaceLineBO(AbstractSolver):
    """
    Bayesian optimization solver that operates in the latent space using line
    Bayesian Optimization [1].

    Parameters
    ----------
    black_box : AbstractBlackBox
        The black box function to be optimized.
    x0 : np.ndarray
        Initial input data.
    y0 : np.ndarray
        Initial output data.
    encoder : Callable[[np.ndarray], np.ndarray]
        Encoder function that maps input data to latent space.
    decoder : Callable[[np.ndarray], np.ndarray]
        Decoder function that maps latent space to input data.
    acq_function : Type[AcquisitionFunction], optional
        The acquisition function to be used, by default ExpectedImprovement.
    bounds : Tuple[float, float], optional
        The bounds of the latent space, by default (-2.0, 2.0).
    type_of_line : str, optional
        The type of line search to be performed, by default "random".

    Attributes
    ----------
    encoder : Callable[[np.ndarray], np.ndarray]
        Encoder function that maps input data to latent space.
    decoder : Callable[[np.ndarray], np.ndarray]
        Decoder function that maps latent space to input data.
    acq_function : Type[AcquisitionFunction]
        The acquisition function to be used.
    bounds : Tuple[float, float]
        The bounds of the latent space.
    type_of_line : str
        The type of line search to be performed.

    Methods
    -------
    next_candidate() -> np.ndarray
        Generates the next candidate point in the latent space.

    References
    ----------
    [1] Kirschner, Johannes, Mojmir Mutny, Nicole Hiller, Rasmus Ischebeck, and Andreas Krause.
        “Adaptive and Safe Bayesian Optimization in High Dimensions via One-Dimensional Subspaces.”
        In Proceedings of the 36th International Conference on Machine Learning, 3429–38. PMLR, 2019.
        https://proceedings.mlr.press/v97/kirschner19a.html.

    """

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
        Initialize the LatentSpaceLineBO solver.

        Parameters
        ----------
        black_box : AbstractBlackBox
            The black box function to be optimized.
        x0 : np.ndarray
            Initial input data.
        y0 : np.ndarray
            Initial output data.
        encoder : Callable[[np.ndarray], np.ndarray]
            Encoder function that maps input data to latent space.
        decoder : Callable[[np.ndarray], np.ndarray]
            Decoder function that maps latent space to input data.
        acq_function : Type[AcquisitionFunction], optional
            The acquisition function to be used, by default ExpectedImprovement.
        bounds : Tuple[float, float], optional
            The bounds of the latent space, by default (-2.0, 2.0).
        type_of_line : str, optional
            The type of line selection to be performed, by default "random".
            Options are: {"random", "coordinate"}

        """
        super().__init__(black_box, x0, y0)
        self.encoder = encoder
        self.decoder = decoder
        self.acq_function = acq_function
        self.bounds = bounds
        self.type_of_line = type_of_line

    def next_candidate(self) -> np.ndarray:
        """
        Generates the next candidate point in the search space.

        Returns
        -------
        np.ndarray
            The next candidate point (not in latent space, but
            in search space).

        """
        # Encode the data to latent space
        x = np.concatenate(self.history["x"], axis=0)
        y = np.concatenate(self.history["y"], axis=0)

        z = self.encoder(x)

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
