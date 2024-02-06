"""This module tests our implementation of NSGA-II."""

from pathlib import Path

import pytest
import numpy as np


def test_running_nsga_ii():
    from poli_baselines.solvers import DiscreteNSGAII
    from poli import objective_factory
    from poli.core.multi_objective_black_box import MultiObjectiveBlackBox

    population_size = 10

    _, f_aloha, x0, y0, _ = objective_factory.create(
        name="aloha",
    )
    alphabet = f_aloha.info.alphabet

    # Randomly choosing 10 5-letter arrays
    x0 = np.random.choice(alphabet, size=(population_size, 5))

    f = MultiObjectiveBlackBox(
        info=f_aloha.info,
        objective_functions=[f_aloha, f_aloha],
    )

    y0 = f(x0)

    solver = DiscreteNSGAII(
        black_box=f,
        x0=x0,
        y0=y0,
        population_size=population_size,
        initialize_with_x0=True,
    )

    solver.solve(max_iter=5, verbose=True)


def test_nsga_ii_in_docs():
    import numpy as np

    from poli.objective_repository import AlohaProblemFactory
    from poli.core.multi_objective_black_box import MultiObjectiveBlackBox

    from poli_baselines.solvers import DiscreteNSGAII

    population_size = 15
    batch_size = 10
    max_iterations = 100
    num_mutations = 1

    # Creating the aloha problem
    problem_factory = AlohaProblemFactory()
    f_aloha, _, _ = problem_factory.create()

    # Putting two copies together to make a multi-objective black box
    f = MultiObjectiveBlackBox(
        info=f_aloha.info,
        objective_functions=[f_aloha, f_aloha],
    )

    # Creating a random initial population
    x0 = np.random.choice(f.info.alphabet, size=(batch_size, 5))
    y0 = f(x0)

    solver = DiscreteNSGAII(
        black_box=f,
        x0=x0,
        y0=y0,
        population_size=population_size,
        num_mutations=num_mutations,
    )

    solver.solve(max_iter=max_iterations)
    print(f"Best solution: {solver.get_best_solution()}")


def test_nsga_ii_on_foldx():
    _ = pytest.importorskip("Bio")

    from poli.objective_repository.foldx_stability_and_sasa.register import (
        FoldXStabilityAndSASAProblemFactory,
    )
    from poli_baselines.solvers import DiscreteNSGAII

    THIS_DIR = Path(__file__).parent.resolve()

    wildtype_pdb_paths = list(
        (THIS_DIR.parent / "simple" / "genetic_algorithm" / "example_pdbs").glob(
            "*_Repair.pdb"
        )
    )

    problem_factory = FoldXStabilityAndSASAProblemFactory()
    f, x0, y0 = problem_factory.create(
        wildtype_pdb_path=wildtype_pdb_paths,
        batch_size=3,
        seed=0,
        parallelize=True,
        num_workers=3,
    )

    solver = DiscreteNSGAII(
        black_box=f,
        x0=x0,
        y0=y0,
        population_size=5,
        num_mutations=1,
    )

    solver.solve(max_iter=5, verbose=True)


if __name__ == "__main__":
    test_nsga_ii_in_docs()
