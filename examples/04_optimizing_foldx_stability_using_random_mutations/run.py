"""
TODO: write.
"""
from pathlib import Path

from poli import objective_factory
from poli_baselines.solvers.simple.random_mutation import RandomMutation

THIS_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    MAX_ITERATIONS = 4

    # This is also known as mRogue.
    WILDTYPE_PDB_PATH = THIS_DIR / "3ned_Repair.pdb"

    # Using the registered factory, we can instantiate our objective
    problem_info, f, x0, y0, run_info = objective_factory.create(
        name="foldx_stability",
        caller_info=None,
        observer=None,
        wildtype_pdb_path=WILDTYPE_PDB_PATH,
    )

    # Let's instantiate a baseline solver
    baseline = RandomMutation(
        black_box=f,
        x0=x0,
        y0=y0,
    )
    baseline.solve(max_iter=MAX_ITERATIONS)
    print(baseline.history)

    baseline.save_history(
        THIS_DIR / f"random_mutations_{WILDTYPE_PDB_PATH.stem}.json",
        alphabet=problem_info.alphabet,
    )
