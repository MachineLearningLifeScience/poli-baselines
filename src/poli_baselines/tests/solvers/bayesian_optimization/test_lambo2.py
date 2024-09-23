from pathlib import Path

from poli.repository import RaspProblemFactory

from poli_baselines.solvers.bayesian_optimization.lambo2 import LaMBO2


TEST_ASSETS = Path(__file__).parent.parent.parent / "test_files"


def test_lambo2_runs_on_rasp():
    problem = RaspProblemFactory().create(
        wildtype_pdb_path=TEST_ASSETS / "2vad_A.pdb",
        chains_to_keep=["A"],
        additive=True,
    )
    f, x0 = problem.black_box, problem.x0

    solver = LaMBO2(
        black_box=f,
        x0=x0,
        y0=f(x0),
    )

    solver.solve(max_iter=1)
