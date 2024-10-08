"""
This solver is meant to be run inside the hdbo__pr env.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from poli.core.abstract_black_box import AbstractBlackBox

from poli_baselines.core.abstract_solver import AbstractSolver
from poli_baselines.core.utils.bo_pr.run_one_replication import (
    run_one_replication_on_poli_black_box,
)


class ProbabilisticReparametrizationSolver(AbstractSolver):
    """
    A bridge between PR and poli-baselines solvers.

    The keyword arguments were selected according to
    the config files in the problems directory.
    """

    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        seed: int | None = None,
        batch_size: int = 1,
        mc_samples: int = 256,
        n_initial_points: int | None = None,
        sequence_length: int | None = None,
        alphabet: list[str] | None = None,
        noise_std: float | None = None,
        use_fixed_noise: bool = False,
        device: torch.device | str | None = None,
        label: Literal[
            "sobol",
            "cont_optim__round_after__ei",
            "pr__ei",
            "exact_round__fin_diff__ei",
            "exact_round__ste__ei",
            "enumerate__ei",
            "cont_optim__round_after__ts",
            "pr__ts",
            "exact_round__fin_diff__ts",
            "exact_round__ste__ts",
            "enumerate__ts",
            "cont_optim__round_after__ucb",
            "pr__ucb",
            "exact_round__fin_diff__ucb",
            "exact_round__ste__ucb",
            "enumerate__ucb",
            "cont_optim__round_after__ehvi",
            "pr__ehvi",
            "exact_round__fin_diff__ehvi",
            "exact_round__ste__ehvi",
            "enumerate__ehvi",
            "cont_optim__round_after__nehvi-1",
            "pr__nehvi-1",
            "exact_round__fin_diff__nehvi-1",
            "exact_round__ste__nehvi-1",
            "enumerate__nehvi-1",
            "nevergrad_portfolio",
        ] = "pr__ei",
    ):
        super().__init__(black_box, x0, y0)
        if x0 is None or x0.size == 0 or y0.size == 0:
            assert (
                n_initial_points is not None
            ), "n_initial_points must be provided if you are not providing initial points."

        sequence_length_ = sequence_length or self.black_box.info.max_sequence_length
        if sequence_length_ is None or sequence_length_ == float("inf"):
            raise ValueError("Sequence length must be provided.")
        self.sequence_length = sequence_length_
        self.device = device
        alphabet_ = alphabet or self.black_box.info.alphabet
        if alphabet_ is None:
            raise ValueError(
                f"For this specific black box ({self.black_box.info.name}), an alphabet must be provided."
            )
        self.alphabet = alphabet_
        self.alphabet_s_to_i = {s: i for i, s in enumerate(self.alphabet)}
        self.alphabet_i_to_s = {i: s for i, s in enumerate(self.alphabet)}

        if isinstance(x0, np.ndarray):
            # Checking that it's of the form [_, L], where
            # L is the sequence length.
            assert x0.ndim == 2
            assert (
                x0.shape[1] == self.sequence_length
            ), "We expect the input x0 to be an array of shape [b, L], where L is the sequence length."

        if seed is None:
            seed = np.random.randint(0, 10_000)
        self.seed = seed
        self.batch_size = batch_size
        self.mc_samples = mc_samples
        self.n_initial_points = n_initial_points
        self.label = label
        self.noise_std = noise_std
        self.use_fixed_noise = use_fixed_noise

    def solve(
        self,
        max_iter: int,
        device: torch.device | str = "cpu",
    ):
        if self.x0 is not None:
            # We need to transform it to a tensor of integers.
            X_init_ = [[self.alphabet_s_to_i[s] for s in x_i] for x_i in self.x0]
            X_init = torch.Tensor(X_init_).long()
            X_init = torch.nn.functional.one_hot(X_init, len(self.alphabet)).flatten(
                start_dim=1
            )
        else:
            X_init = None

        if self.y0 is None:
            Y_init = None
        else:
            Y_init = torch.from_numpy(self.y0)

        run_one_replication_on_poli_black_box(
            seed=self.seed,
            label=self.label,
            iterations=max_iter,
            black_box=self.black_box,
            batch_size=self.batch_size,
            mc_samples=self.mc_samples,
            n_initial_points=self.n_initial_points,
            problem_kwargs={
                "sequence_length": self.sequence_length,
                "alphabet": self.alphabet,
                "negate": False,
                "noise_std": self.noise_std,
            },
            model_kwargs={
                "use_fixed_noise": self.use_fixed_noise,
            },
            save_callback=lambda t: t,
            device=self.device if self.device else device,
            X_init=X_init,
            Y_init=Y_init,
        )
