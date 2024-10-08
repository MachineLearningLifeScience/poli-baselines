"""
This module tests the implementation of a genetic algorithm inside poli-baselines.
"""


def test_genetic_algorithm_interface():
    from poli.objective_repository import AlohaProblemFactory

    from poli_baselines.solvers.simple.genetic_algorithm import (
        FixedLengthGeneticAlgorithm,
    )

    problem = AlohaProblemFactory().create()
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)

    solver = FixedLengthGeneticAlgorithm(black_box=f, x0=x0, y0=y0, population_size=10)

    solver.solve(max_iter=100)


def test_genetic_algorithm_improves_over_time():
    from poli.core.util.seeding import seed_python_numpy_and_torch
    from poli.objective_repository import AlohaProblemFactory

    from poli_baselines.solvers.simple.genetic_algorithm import (
        FixedLengthGeneticAlgorithm,
    )

    seed_python_numpy_and_torch(13)

    problem = AlohaProblemFactory().create()
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)

    solver = FixedLengthGeneticAlgorithm(black_box=f, x0=x0, y0=y0, population_size=10)

    solver.solve(max_iter=20)
    assert y0 < solver.get_best_performance()
