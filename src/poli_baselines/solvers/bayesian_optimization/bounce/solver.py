"""
This can run inside the bounce env.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch

from poli.core.abstract_black_box import AbstractBlackBox
from poli_baselines.core.abstract_solver import AbstractSolver

ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.parent.resolve()

from bounce.benchmarks import PoliBenchmark
from bounce.bounce import Bounce


class BounceSolver(AbstractSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        noise_std: float = 0.0,
        sequence_length: int | None = None,
        alphabet: dict[str, int] | None = None,
        device: str | None = None,
        dtype: Literal["float32", "float64"] = "float32",
        batch_size: int = 1,
        number_initial_points: int = 2,
        initial_target_dimensionality: int = 2,
        number_new_bins_on_split: int = 2,
    ):
        super().__init__(black_box, x0, y0)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.number_initial_points = number_initial_points
        self.initial_target_dimensionality = initial_target_dimensionality
        self.number_new_bins_on_split = number_new_bins_on_split

        self.benchmark = PoliBenchmark(
            f=black_box,
            noise_std=noise_std,
            sequence_length=sequence_length,
            alphabet=alphabet,
        )

        # Creating the results dir for bounce
        results_dir = ROOT_DIR / "bounce_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Creating a gitignore file inside that dir
        with open(results_dir / ".gitignore", "w") as fp:
            fp.write("*\n!.gitignore")

    def solve(self, max_iter: int) -> None:
        self.bounce = Bounce(
            benchmark=self.benchmark,
            number_initial_points=self.number_initial_points,
            initial_target_dimensionality=self.initial_target_dimensionality,
            number_new_bins_on_split=self.number_new_bins_on_split,
            maximum_number_evaluations=max_iter,
            batch_size=self.batch_size,
            results_dir=str(ROOT_DIR / "data" / "bounce_results"),
            device=self.device,
            dtype=self.dtype,
        )
        self.bounce.run()
