"""Tests adding callbacks to solvers."""

from unittest.mock import MagicMock

import numpy as np
from poli.objective_repository.toy_continuous_problem.register import (
    ToyContinuousProblemFactory,
)

from poli_baselines.solvers.simple.continuous_random_mutation import (
    ContinuousRandomMutation,
)


class TestAddingCallbacks:
    problem = ToyContinuousProblemFactory().create(
        function_name="ackley_function_01", n_dimensions=10
    )
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)

    solver = ContinuousRandomMutation(
        black_box=f,
        x0=np.random.randn(10).reshape(1, -1),
        y0=np.random.randn(1).reshape(1, -1),
    )

    def test_adding_a_pre_step_callback(self):
        mock = MagicMock()
        self.solver.solve(
            max_iter=2,
            pre_step_callbacks=[lambda solver: mock(solver)],
        )

        assert mock.call_count == 2

    def test_adding_a_post_step_callback(self):
        mock = MagicMock()
        self.solver.solve(
            max_iter=2,
            post_step_callbacks=[lambda solver: mock(solver)],
        )

        assert mock.call_count == 2
