"""This module implements Thompson Sampling (TS) as an acquisition function inside BoTorch."""

from typing import Tuple

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.generation import MaxPosteriorSampling
from botorch.models.model import Model
from torch.quasirandom import SobolEngine


class ThompsonSampling(AcquisitionFunction):
    """Thompson Sampling acquisition function.

    This acquisition function is used to implement Thompson Sampling
    as an acquisition function inside BoTorch. It is used to sample
    the next point to evaluate in the trust region.

    This implementation adapts the code from the "create_candidate"
    function in the BoTorch tutorial on BAxUS [1].

    Parameters
    ----------
    model : Model
        The model to use for the acquisition function.
    trust_region : Tuple[torch.Tensor, torch.Tensor]
        The trust region to sample from.
    n_candidates : int, optional
        The number of candidates to sample from the trust region.
        Defaults to 100.

    References
    ----------
    [1] https://botorch.org/tutorials/baxus

    Notes
    -----
    - Even though we inherit from AcquisitionFunction, the
      "forward" method is slightly different: it takes as input
      the dimensionality of the problem.
    """

    def __init__(
        self,
        model: Model,
        trust_region: Tuple[torch.Tensor, torch.Tensor],
        n_candidates: int = 100,
    ) -> None:
        """Initializes the Thompson Sampling acquisition function.

        Parameters
        ----------
        model : Model
            The model to use for the acquisition function.
        trust_region : Tuple[torch.Tensor, torch.Tensor]
            The trust region to sample from.
        n_candidates : int, optional
            The number of candidates to sample from the trust region.
            Defaults to 100.
        """
        super().__init__(model)
        self.n_candidates = n_candidates
        self.trust_region = trust_region

    def compute_center_from_trust_region(self, tr_lb, tr_ub) -> torch.Tensor:
        """Computes the center of the current trust region."""
        return (tr_lb + tr_ub) / 2.0

    def forward(self, dim: int) -> torch.Tensor:
        """Computes the next point to evaluate using Thompson Sampling.

        This implementation is taken from the "create_candidate" function
        in the BoTorch tutorial on BAxUS [1].

        Parameters
        ----------
        dim : int
            The dimensionality of the problem.

        References
        ----------
        [1] https://botorch.org/tutorials/baxus
        """
        tr_lb, tr_ub = self.trust_region

        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(self.n_candidates).to(dtype=tr_lb.dtype, device=tr_lb.device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        x_center = self.compute_center_from_trust_region(tr_lb, tr_ub)

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (
            torch.rand(
                self.n_candidates, dim, dtype=x_center.dtype, device=x_center.device
            )
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim, size=(len(ind),), device=x_center.device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(self.n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=self.model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=1)

        return X_next
