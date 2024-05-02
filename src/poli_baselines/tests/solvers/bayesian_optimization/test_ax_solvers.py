"""Testing all Ax-related solvers.

This test suite runs only on poli__ax in CI.
"""

import pytest

from poli.objective_repository import ToyContinuousProblemFactory

from poli_baselines.core.abstract_solver import AbstractSolver

try:
    import ax  # type: ignore[import]
except ImportError:
    pytest.skip("Ax is not installed.", allow_module_level=True)

from poli_baselines.solvers.bayesian_optimization.saasbo import SAASBO
from poli_baselines.solvers.bayesian_optimization.alebo import ALEBO
from poli_baselines.solvers.bayesian_optimization.vanilla_bo_hvarfner import (
    VanillaBOHvarfner,
)


@pytest.mark.parametrize(
    "solver, solver_kwargs",
    [
        [SAASBO, {}],
        [ALEBO, {"lower_dim": 2}],
        [VanillaBOHvarfner, {}],
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
        solver: AbstractSolver = solver(
            black_box=f,
            x0=x0,
            y0=y0,
            **solver_kwargs,
        )

        solver.solve(max_iter=2)
