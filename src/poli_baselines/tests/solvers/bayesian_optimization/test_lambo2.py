import importlib
from pathlib import Path
from unittest.mock import patch

import pytest

import numpy as np

from poli.repository import AlohaProblemFactory

if importlib.util.find_spec("cortex") is None:
    pytest.skip("Cortex is not installed.", allow_module_level=True)

TEST_ASSETS = Path(__file__).parent.parent.parent / "test_files"


def test_lambo2_runs_on_mocked_function():
    from poli_baselines.solvers.bayesian_optimization.lambo2 import LaMBO2

    problem = AlohaProblemFactory().create()
    f, x0 = problem.black_box, problem.x0

    with patch.object(f, "__call__", new=lambda x: np.zeros((x.shape[0], 1))) as _:
        solver = LaMBO2(
            black_box=f,
            x0=x0,
            y0=f(x0),
        )

        solver.solve(max_iter=1)
