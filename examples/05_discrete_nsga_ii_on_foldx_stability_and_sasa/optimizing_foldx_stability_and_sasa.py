"""Using DiscreteNSGAII to optimize foldx stability and SASA.

By default, the DiscreteNSGAII solver uses a DiscreteSequenceMating,
which copies the population and mutates each offspring at a random position
self.num_mutations times.

This example requires you to have foldx installed. We expect
- the binary to be at ~/foldx/foldx
- (if you are using foldx v4) the rotabase.txt file to be at ~/foldx/rotabase.txt

For more information, check the documentation of the `foldx_stability` objective.

The PDB files were taken from LaMBO's assets [1].

[1] Accelerating Bayesian Optimization for Protein Design with Denoising Autoencoders
    by Stanton et al. (2022). https://github.com/samuelstanton/lambo
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from poli.objective_repository import FoldXStabilityAndSASAProblemFactory

from poli_baselines.solvers import DiscreteNSGAII

THIS_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    wildtype_pdb_paths = (THIS_DIR / "pdbs").glob("**/*_Repair.pdb")
    wildtype_pdb_paths = list(wildtype_pdb_paths)

    problem_factory = FoldXStabilityAndSASAProblemFactory()

    f, x0, y0 = problem_factory.create(
        wildtype_pdb_path=wildtype_pdb_paths,
        parallelize=True,
        num_workers=6,
    )

    solver = DiscreteNSGAII(
        black_box=f,
        x0=x0,
        y0=y0,
        population_size=10,
        initialize_with_x0=True,
        num_mutations=5,
    )

    _, ax = plt.subplots(1, 1)
    for _ in range(10):
        ax.clear()
        if len(solver.history["y"]) > 0:
            all_previous_y = np.concatenate(solver.history["y"], axis=0)
            ax.scatter(
                all_previous_y[:, 0], all_previous_y[:, 1], color="gray", alpha=0.5
            )
        population, fitnesses = solver.step()

        print(population)
        print(fitnesses)
        ax.scatter(fitnesses[:, 0], fitnesses[:, 1], color="red")
        plt.pause(0.1)

    solver.save_history(THIS_DIR / "history.json")
