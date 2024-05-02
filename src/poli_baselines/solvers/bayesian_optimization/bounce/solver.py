"""
This can run inside the bounce env.
"""

import numpy as np

from poli.objective_repository import AlbuterolSimilarityBlackBox  # type: ignore[import]
from poli.core.abstract_black_box import AbstractBlackBox  # type: ignore[import]
from poli_baselines.core.abstract_solver import AbstractSolver  # type: ignore[import]

from bounce.benchmarks import PoliBenchmark  # type: ignore[import]
from bounce.bounce import Bounce  # type: ignore[import]

from hdbo_benchmark.utils.constants import ROOT_DIR, DEVICE


class BounceSolver(AbstractSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        noise_std: float = 0.0,
        sequence_length: int | None = None,
        alphabet: dict[str, int] | None = None,
    ):
        super().__init__(black_box, x0, y0)
        self.benchmark = PoliBenchmark(
            f=black_box,
            noise_std=noise_std,
            sequence_length=sequence_length,
            alphabet=alphabet,
        )

        # Creating the results dir for bounce
        results_dir = ROOT_DIR / "data" / "bounce_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Creating a gitignore file inside that dir
        with open(results_dir / ".gitignore", "w") as fp:
            fp.write("*\n!.gitignore")

    def solve(self, max_iter: int) -> None:
        self.bounce = Bounce(
            benchmark=self.benchmark,
            number_initial_points=2,
            initial_target_dimensionality=2,
            number_new_bins_on_split=2,
            maximum_number_evaluations=max_iter,
            batch_size=1,
            results_dir=str(ROOT_DIR / "data" / "bounce_results"),
            device=DEVICE,
            dtype="float32",
        )
        self.bounce.run()
