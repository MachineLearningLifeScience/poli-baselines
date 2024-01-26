"""
This module tests the implementation of a genetic algorithm inside poli-baselines.
"""
from pathlib import Path

import pytest


def test_genetic_algorithm_interface():
    from poli.objective_repository import AlohaProblemFactory
    from poli_baselines.solvers import GeneticAlgorithm

    f, x0, y0 = AlohaProblemFactory().create()

    solver = GeneticAlgorithm(black_box=f, x0=x0, y0=y0, pop_size=10)

    solver.solve(max_iter=100)


def test_genetic_algorithm_improves_over_time():
    from poli.objective_repository import AlohaProblemFactory
    from poli_baselines.solvers import GeneticAlgorithm

    f, x0, y0 = AlohaProblemFactory().create()

    solver = GeneticAlgorithm(black_box=f, x0=x0, y0=y0, pop_size=10)

    solver.solve(max_iter=10)
    assert y0 < solver.get_best_performance()


def test_genetic_algorithm_on_foldx_stability_with_several_wildtypes():
    from poli_baselines.solvers import GeneticAlgorithm

    register = pytest.importorskip("poli.objective_repository.foldx_stability.register")

    THIS_DIR = Path(__file__).parent.resolve()

    wildtype_pdb_paths = list((THIS_DIR / "example_pdbs").glob("*_Repair.pdb"))

    problem_factory = register.FoldXStabilityProblemFactory()
    f, x0, y0 = problem_factory.create(
        wildtype_pdb_path=wildtype_pdb_paths, verbose=True
    )

    solver = GeneticAlgorithm(black_box=f, x0=x0, y0=y0, pop_size=5)
    solver.solve(max_iter=2)
    print(solver.get_best_solution())


if __name__ == "__main__":
    test_genetic_algorithm_on_foldx_stability_with_several_wildtypes()
