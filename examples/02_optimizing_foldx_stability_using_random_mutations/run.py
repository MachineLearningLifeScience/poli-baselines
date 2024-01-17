"""
In this example, we optimize the stability of a protein (i.e. -ddG)
using random mutations of the wildtype.

This example requires you to have foldx installed. We expect
- the binary to be at ~/foldx/foldx
- (if you are using foldx v4) the rotabase.txt file to be at ~/foldx/rotabase.txt

For more information, check the documentation of the `foldx_stability` objective.
"""
from pathlib import Path

from poli import objective_factory
from poli_baselines.solvers.simple.random_mutation import RandomMutation

THIS_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    MAX_ITERATIONS = 4

    # This is also known as mRouge.
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
