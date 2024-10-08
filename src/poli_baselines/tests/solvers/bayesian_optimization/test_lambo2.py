import importlib
from pathlib import Path

import pytest
from poli.repository import EhrlichProblemFactory

if importlib.util.find_spec("cortex") is None:
    pytest.skip("Cortex is not installed.", allow_module_level=True)

TEST_ASSETS = Path(__file__).parent.parent.parent / "test_files"


def test_lambo2_runs_on_ehrlich():
    from poli_baselines.solvers.bayesian_optimization.lambo2 import LaMBO2

    problem = EhrlichProblemFactory().create(
        sequence_length=10,
        n_motifs=2,
        motif_length=4,
        return_value_on_unfeasible=-10.0,
    )
    f, x0 = problem.black_box, problem.x0

    solver = LaMBO2(
        black_box=f,
        x0=x0,
        y0=f(x0),
        overrides=["max_epochs=1"],
    )

    solver.solve(max_iter=1)


if __name__ == "__main__":
    test_lambo2_runs_on_ehrlich()
