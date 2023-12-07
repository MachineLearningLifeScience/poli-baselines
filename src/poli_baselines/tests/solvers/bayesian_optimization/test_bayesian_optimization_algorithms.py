"""Tests all Bayesian Optimization algorithms together under the same class"""
import pytest
from unittest.mock import patch

import torch
from botorch.models import SingleTaskGP

from poli.objective_repository.toy_continuous_problem.register import (
    ToyContinuousProblemFactory,
)

from poli_baselines.solvers import LineBO, SAASBO, VanillaBayesianOptimization
from poli_baselines.solvers.bayesian_optimization.base_bayesian_optimization import (
    BaseBayesianOptimization,
)


@pytest.mark.parametrize(
    "solver, solver_kwargs",
    [
        [LineBO, {"type_of_line": "random"}],
        [SAASBO, {}],
        [VanillaBayesianOptimization, {}],
    ],
)
class TestBayesianOptimization:
    arr = ToyContinuousProblemFactory().create(
        function_name="ackley_function_01", n_dimensions=10
    )
    model = SingleTaskGP(
        train_X=torch.from_numpy(arr[1]),
        train_Y=torch.from_numpy(arr[2]),
    )

    def test_instancing(self, solver, solver_kwargs):
        f, x0, y0 = self.arr
        solver(
            black_box=f,
            x0=x0,
            y0=y0,
            **solver_kwargs,
        )

    def test_solving_while_mocking_training(self, solver, solver_kwargs):
        f, x0, y0 = self.arr
        solver: BaseBayesianOptimization = solver(
            black_box=f,
            x0=x0,
            y0=y0,
            **solver_kwargs,
        )

        with patch.object(solver, "_fit_model", return_value=self.model) as mock_method:
            solver.solve(
                max_iter=2,
            )

        assert mock_method.call_count == 2
