"""
This module tests the implementation of a genetic algorithm inside poli-baselines.
"""


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


if __name__ == "__main__":
    test_genetic_algorithm_improves_over_time()
