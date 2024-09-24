"""
This can run inside the corel env.
"""

from __future__ import annotations
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from poli.core.util.seeding import seed_python_numpy_and_torch
from poli.core.abstract_black_box import AbstractBlackBox
from poli_baselines.core.abstract_solver import AbstractSolver
from poli_baselines.core.utils.bounce.poli_benchmark import PoliBenchmark

from corel.corel import Corel # TODO make corel installable
from corel.weightings import AbstractWeighting

ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.parent.resolve()


class CorelSolver(AbstractSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        weighting: AbstractWeighting, 
        x0: np.ndarray = None,
        y0: np.ndarray = None,
        noise_std: float = 0.0,
        sequence_length: int | None = None,
        alphabet: list[str] | None = None,
        device: str | None = None,
        dtype: Literal["float32", "float64"] = "float32",
        batch_size: int = 1,
        n_initial_points: int | None = None,
        initial_target_dimensionality: int = 2,
        number_new_bins_on_split: int = 2,
    ):
        super().__init__(black_box, None, None)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if weighting is None:
            raise RuntimeError("CoRel requires a weighting to run!\nProvide an {LVM | HMM} for the problem.")
        self.x0 = x0
        self.y0 = y0
        self.weighting = weighting
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.initial_target_dimensionality = initial_target_dimensionality
        self.number_new_bins_on_split = number_new_bins_on_split
        self.n_initial_points = n_initial_points

        self.benchmark = PoliBenchmark(
            f=black_box,
            noise_std=noise_std,
            sequence_length=sequence_length,
            alphabet=alphabet,
        )

        # Creating the results dir for bounce
        results_dir = ROOT_DIR / "corel_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Creating a gitignore file inside that dir
        with open(results_dir / ".gitignore", "w") as fp:
            fp.write("*\n!.gitignore")

    def solve(
        self,
        max_iter: int = 100,
        n_initial_points: int | None = None,
        seed: int | None = None,
    ) -> None:
        if seed is not None:
            seed_python_numpy_and_torch(seed)

        if n_initial_points is None:
            if self.n_initial_points is None:
                raise ValueError(
                    "n_initial_points must be set, either in init or in solve"
                )
            n_initial_points = self.n_initial_points

        self.corel = Corel(
            benchmark=self.benchmark, # TODO: check benchmark to black_box refactor
            x0=self.x0,
            y0=self.y0,
            weighting=self.weighting,
            initial_target_dimensionality=self.initial_target_dimensionality,
            number_new_bins_on_split=self.number_new_bins_on_split,
            maximum_number_evaluations=max_iter,
            batch_size=self.batch_size,
            results_dir=str(ROOT_DIR / "data" / "corel_results"),
            device=self.device,
            dtype=self.dtype,
        )
        self.corel.run()
