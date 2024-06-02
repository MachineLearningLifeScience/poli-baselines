import numpy as np

import torch
import gpytorch  # type: ignore[import]
from gpytorch.constraints.constraints import GreaterThan  # type: ignore[import]
from botorch.models import SingleTaskGP  # type: ignore[import]
from botorch.acquisition.logei import qLogNoisyExpectedImprovement  # type: ignore[import]

from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep  # type: ignore[import]
from ax.modelbridge.registry import Models  # type: ignore[import]
from ax.models.torch.botorch_modular.surrogate import Surrogate  # type: ignore[import]

from poli.core.abstract_black_box import AbstractBlackBox  # type: ignore[import]

from poli_baselines.core.utils.ax.ax_solver import AxSolver


class VanillaBOHvarfner(AxSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        bounds: list[tuple[float, float]] | None = None,
        noise_std: float = 0,
        device: torch.device | str | None = None,
    ):
        _, n_dimensions = x0.shape
        generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.BOTORCH_MODULAR,
                    num_trials=-1,
                    max_parallelism=black_box.num_workers,
                    model_kwargs={
                        "surrogate": Surrogate(
                            botorch_model_class=SingleTaskGP,
                            covar_module_class=gpytorch.kernels.RBFKernel,
                            covar_module_options={
                                "lengthscale_prior": gpytorch.priors.LogNormalPrior(
                                    1.4 + np.log(n_dimensions) / 2, 1.73205
                                ),
                                "lengthscale_constraint": GreaterThan(1e-4),
                            },
                            likelihood_class=gpytorch.likelihoods.GaussianLikelihood,
                            likelihood_options={
                                "noise_prior": gpytorch.priors.LogNormalPrior(
                                    -4.0, 1.0
                                ),
                                "noise_constraint": GreaterThan(1e-4),
                            },
                        ),
                        "botorch_acqf_class": qLogNoisyExpectedImprovement,
                        "acquisition_options": {
                            "prune_baseline": True,
                        },
                        "device": device,
                    },
                )
            ]
        )
        super().__init__(black_box, x0, y0, generation_strategy, bounds, noise_std)


if __name__ == "__main__":
    from poli.objective_repository import ToyContinuousBlackBox  # type: ignore[import]

    black_box = ToyContinuousBlackBox(
        function_name="branin_2d",
        n_dimensions=2,
        embed_in=50,
        dimensions_to_embed_in=[19, 34],
    )
    x0 = np.random.uniform(-1.0, 1.0, (10, 50))
    y0 = black_box(x0)

    vanilla_bo_solver = VanillaBOHvarfner(
        black_box=black_box,
        x0=x0,
        y0=y0,
    )
    vanilla_bo_solver.solve(max_iter=50, verbose=True)
