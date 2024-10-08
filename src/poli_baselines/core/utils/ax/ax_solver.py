from __future__ import annotations

import uuid
from typing import Tuple

import numpy as np
import torch
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.service.ax_client import AxClient, ObjectiveProperties
from numpy import ndarray
from poli.core.abstract_black_box import AbstractBlackBox
from poli.objective_repository import ToyContinuousBlackBox

from poli_baselines.core.abstract_solver import AbstractSolver
from poli_baselines.core.utils.ax.interface import define_search_space


class AxSolver(AbstractSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: ndarray,
        y0: ndarray,
        generation_strategy: GenerationStrategy,
        bounds: list[tuple[float, float]] | tuple[float, float] | None = None,
        noise_std: float = 0.0,
        device: torch.device | None = None,
    ):
        super().__init__(black_box, x0, y0)
        self.noise_std = noise_std

        if bounds is None:
            assert isinstance(black_box, ToyContinuousBlackBox)
            bounds_ = [black_box.function.limits] * x0.shape[1]
        else:
            # If bounds is (lb, up), then we build the bounds
            # for the user
            if len(bounds) == 2:
                if isinstance(bounds[0], (int, float)):
                    assert isinstance(
                        bounds[1], (int, float)
                    ), f"Expected a float for the upper bound. Got bounds={bounds}"
                    bounds_ = [bounds] * x0.shape[1]
                else:
                    assert isinstance(
                        bounds[0], (tuple, list)
                    ), f"Expected a tuple or list for bounds. Got bounds={bounds}"
                    bounds_ = bounds
            else:
                bounds_ = bounds

            assert len(bounds_) == x0.shape[1]
            assert all(len(bound) == 2 for bound in bounds_)

        ax_client = AxClient(
            generation_strategy=generation_strategy, torch_device=device
        )
        exp_id = f"{uuid.uuid4()}"[:8]

        search_space = define_search_space(x0=x0, bounds=bounds_)

        ax_client.create_experiment(
            name=f"experiment_on_{black_box.info.name}_{exp_id}",
            parameters=[
                {
                    "name": param.name,
                    "type": "range",
                    "bounds": [param.lower, param.upper],
                    "value_type": "float",
                }
                for param in search_space.parameters.values()
            ],
            objectives={black_box.info.name: ObjectiveProperties(minimize=False)},
        )

        def evaluate(
            parametrization: dict[str, float]
        ) -> dict[str, tuple[float, float]]:
            x = np.array([[parametrization[f"x{i}"] for i in range(x0.shape[1])]])
            y = black_box(x)
            return {black_box.info.name: (y.flatten()[0], self.noise_std)}

        self.evaluate = evaluate

        # Run initialization with x0 and y0
        for x, y in zip(x0, y0):
            params = {f"x{i}": float(x_i) for i, x_i in enumerate(x)}
            _, trial_index = ax_client.attach_trial(params)
            ax_client.complete_trial(
                trial_index=trial_index,
                raw_data={black_box.info.name: (y[0], self.noise_std)},
            )

        print(ax_client.get_trials_data_frame())
        self.ax_client = ax_client

    def solve(
        self,
        max_iter: int = 100,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        for i in range(max_iter):
            parameters, trial_index = self.ax_client.get_next_trial()
            val = self.evaluate(parameters)
            self.ax_client.complete_trial(
                trial_index=trial_index,
                raw_data=val,
            )
            # df = self.ax_client.get_trials_data_frame()

            if verbose:
                print(
                    f"Iteration: {i}, Value in iteration: {val[self.black_box.info.name][0]:.3f}, Best so far: {self.ax_client.get_trials_data_frame()[self.black_box.info.name].max():.3f}"
                )

        # TODO: fix this return
        return self.ax_client.get_trials_data_frame()  # type: ignore
