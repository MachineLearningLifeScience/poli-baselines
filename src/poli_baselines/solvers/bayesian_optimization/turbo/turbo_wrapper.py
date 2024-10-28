# Code taken from https://botorch.org/tutorials/turbo_1
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from poli.core.abstract_black_box import AbstractBlackBox
from torch.quasirandom import SobolEngine

from poli_baselines.core.step_by_step_solver import StepByStepSolver

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DTYPE = torch.get_default_dtype()


NUM_RESTARTS = 10
RAW_SAMPLES = 512


class Turbo(StepByStepSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        bounds: np.ndarray | None = None,
        device: torch.device = DEFAULT_DEVICE,
    ):
        """

        Parameters
        ----------
        black_box
        x0
        y0
        bounds:
            array of shape Dx2 where D is the dimensionality
            The first row contains the lower bounds on x, the last row contains the upper bounds.
        """
        super().__init__(black_box, x0, y0)
        # assert x0.shape[0] > 1

        if bounds is None:
            bounds = np.array([[x0.min() - 1.0, x0.max() + 1.0]] * x0.shape[1])
        if isinstance(bounds, (list, tuple)):
            bounds = np.array([bounds] * x0.shape[1])

        assert bounds.shape[1] == 2
        assert bounds.shape[0] == x0.shape[1]
        assert np.all(bounds[:, 1] >= bounds[:, 0])
        bounds[:, 1] -= bounds[:, 0]

        def make_transforms():
            def to_turbo(X):
                return (X - bounds[:, 0]) / bounds[:, 1]

            def from_turbo(X):
                return X * bounds[:, 1] + bounds[:, 0]

            return to_turbo, from_turbo

        self.device = device
        self.to_turbo, self.from_turbo = make_transforms()
        self.X_turbo = (
            torch.tensor(self.to_turbo(x0))
            .to(self.device)
            .to(torch.get_default_dtype())
        )
        self.Y_turbo = torch.tensor(y0).to(self.device).to(torch.get_default_dtype())
        self.batch_size = 1
        dim = x0.shape[1]
        self.state = TurboState(dim, batch_size=self.batch_size)

    def next_candidate(self) -> np.ndarray:
        dim = self.X_turbo.shape[1]
        # Normalize the data
        # but only if self.Y_turbo.std() is not zero
        if self.Y_turbo.std() > 0:
            train_Y = (self.Y_turbo - self.Y_turbo.mean()) / self.Y_turbo.std()
        else:
            train_Y = self.Y_turbo
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = (
            ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=dim,
                    lengthscale_constraint=Interval(0.005, 4.0),
                )
            )
        )
        model = SingleTaskGP(
            self.X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(DEFAULT_DEVICE)

        fit_gpytorch_mll(mll)

        N_CANDIDATES = min(5000, max(2000, 200 * dim))
        # Create a batch
        X_next = generate_batch(
            state=self.state,
            model=model,
            X=self.X_turbo,
            Y=train_Y,
            batch_size=self.batch_size,
            n_candidates=N_CANDIDATES,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            acqf="ts",
            device=self.device,
        )
        return self.from_turbo(X_next.numpy(force=True))

    def post_update(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        This method is called after the history is updated.
        """
        Y_next = torch.tensor(y).to(self.device)
        X_next = torch.tensor(self.to_turbo(x)).to(self.device)

        # Update state
        self.state = update_state(state=self.state, Y_next=Y_next)
        self.X_turbo = torch.cat((self.X_turbo, X_next), dim=0)
        self.Y_turbo = torch.cat((self.Y_turbo, Y_next), dim=0)


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=512,
    acqf="ts",  # "ei" or "ts"
    device=DEFAULT_DEVICE,
):
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0).to(
        dtype=DEFAULT_DTYPE, device=device
    )
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0).to(
        dtype=DEFAULT_DTYPE, device=device
    )

    if acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=DEFAULT_DTYPE, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (
            torch.rand(n_candidates, dim, dtype=DEFAULT_DTYPE, device=device)
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone().to(device).to(DEFAULT_DTYPE)
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=batch_size)

    elif acqf == "ei":
        ei = qExpectedImprovement(model, Y.max())
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return X_next
