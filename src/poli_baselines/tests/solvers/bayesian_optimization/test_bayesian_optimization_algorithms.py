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
    problem = ToyContinuousProblemFactory().create(
        function_name="ackley_function_01", n_dimensions=10
    )

    def test_instancing(self, solver, solver_kwargs):
        f, x0 = self.problem.black_box, self.problem.x0
        y0 = f(x0)
        solver(
            black_box=f,
            x0=x0,
            y0=y0,
            **solver_kwargs,
        )

    def test_solving_while_mocking_training(self, solver, solver_kwargs):
        f, x0 = self.problem.black_box, self.problem.x0
        y0 = f(x0)
        solver: BaseBayesianOptimization = solver(
            black_box=f,
            x0=x0,
            y0=y0,
            **solver_kwargs,
        )

        model = SingleTaskGP(
            train_X=torch.from_numpy(x0),
            train_Y=torch.from_numpy(y0),
        )
        with patch.object(solver, "_fit_model", return_value=model) as mock_method:
            solver.solve(
                max_iter=2,
            )

        assert mock_method.call_count == 2


def test_documentation_of_bo():
    import numpy as np

    from poli.objective_repository import ToyContinuousBlackBox

    from poli_baselines.solvers import VanillaBayesianOptimization

    f_ackley = ToyContinuousBlackBox(function_name="ackley_function_01", n_dimensions=2)

    x0 = np.random.randn(2).reshape(1, -1).clip(-2.0, 2.0)
    y0 = f_ackley(x0)

    bo_solver = VanillaBayesianOptimization(
        black_box=f_ackley,
        x0=x0,
        y0=y0,
    )

    bo_solver.solve(max_iter=10)
    print(bo_solver.get_best_solution())


if __name__ == "__main__":
    test_documentation_of_bo()
