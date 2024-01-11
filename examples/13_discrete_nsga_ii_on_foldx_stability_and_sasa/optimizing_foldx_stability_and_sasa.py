"""Using DiscreteNSGAII to optimize foldx stability and SASA."""
from pathlib import Path

from pymoo.core.variable import Choice


from poli.objective_repository import FoldXStabilityAndSASAProblemFactory

from poli_baselines.solvers import DiscreteNSGAII
from poli_baselines.core.utils.pymoo.no_crossover import NoCrossover

THIS_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    wildtype_pdb_paths = (THIS_DIR / "pdbs").glob("**/*_Repair.pdb")
    wildtype_pdb_paths = list(wildtype_pdb_paths)[:2]

    problem_factory = FoldXStabilityAndSASAProblemFactory()

    f, x0, y0 = problem_factory.create(
        wildtype_pdb_path=wildtype_pdb_paths,
    )

    solver = DiscreteNSGAII(
        black_box=f,
        x0=x0,
        y0=y0,
        population_size=5,
        initialize_with_x0=True,
    )

    for _ in range(3):
        population, fitnesses = solver.step()

        print(population)
        print(fitnesses)

    solver.save_history(THIS_DIR / "history.json")
