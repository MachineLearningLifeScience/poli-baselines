"""
This has to be run inside the poli__boss environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch

try:
    from boss.code.emukit_models.emukit_ssk_model import SSK_model
    from boss.code.parameters.candidate_parameter import CandidateStringParameter
    from emukit.core import ParameterSpace
    from emukit.core.loop import FixedIterationsStoppingCondition
    from emukit.core.optimization import RandomSearchAcquisitionOptimizer
    from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
    from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
    from emukit.core.initial_designs import RandomDesign
except ImportError as e:
    raise ImportError(
        "You are trying to use the BOSS solver. Install "
        "the relevant optional dependencies with [boss]. \n"
        "You can do this by running: \n"
        "pip install 'poli-baselines[boss] @ git+https://github.com/MachineLearningLifeScience/poli-baselines.git'"
    ) from e

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.util.seeding import seed_python_numpy_and_torch

from poli_baselines.core.abstract_solver import AbstractSolver

ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.parent.resolve()


class BossSolver(AbstractSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray = None,
        y0: np.ndarray = None,
        device: str | None = None,
        dtype: Literal["float32", "float64"] = "float32",
        batch_size: int = 1,
        n_initial_points: int | None = None,
        number_new_bins_on_split: int = 2,
        results_dir: Path | None = None,
    ):
        super().__init__(black_box, None, None)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.x0 = x0
        self.y0 = y0
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.number_new_bins_on_split = number_new_bins_on_split
        self.n_initial_points = n_initial_points

        self.objective = lambda x: -self.black_box(x)  # BOSS minimizes

        # see SMILES examples
        token_space = np.array([" ".join(list(ss)) for ss in self.x0]).reshape(-1,1)

        self.search_space = ParameterSpace([CandidateStringParameter("string", token_space)])  # x0 goes here with correct wrapper
        self.model = SSK_model(self.search_space, self.x0, self.y0, max_subsequence_length=5, n_restarts=1)
        self.acquisition = ExpectedImprovement(self.model)
        self.optimizer = RandomSearchAcquisitionOptimizer(self.search_space, 100)

        self.bo_loop_ssk = BayesianOptimizationLoop(
            model=self.model,
            space=self.search_space,
            acquisition=self.acquisition,
            acquisition_optimizer=self.optimizer,
        )

        # Creating the results dir for boss
        if results_dir is None:
            results_dir = ROOT_DIR / "boss_results"
        Path(results_dir).mkdir(parents=True, exist_ok=True)

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

        stopping_condition = FixedIterationsStoppingCondition(i_max=max_iter)

        self.boss = BossSolver(
            black_box=self.black_box,
            x0=self.x0,
            y0=self.y0,
            n_initial_points=n_initial_points,
            batch_size=self.batch_size,
            results_dir=ROOT_DIR / "data" / "boss_results",
            device=self.device,
            dtype=self.dtype,
        )
        self.boss.bo_loop_ssk.run_loop(self.objective, stopping_condition)

