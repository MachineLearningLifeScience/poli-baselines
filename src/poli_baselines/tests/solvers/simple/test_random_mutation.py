"""This module tests the random mutation solver"""


def test_random_mutation_instantiates_and_runs_well():
    from poli_baselines.solvers import RandomMutation
    from poli import objective_factory

    _, f, x0, y0, _ = objective_factory.create(
        name="aloha",
    )

    solver = RandomMutation(
        black_box=f,
        x0=x0,
        y0=y0,
    )

    solver.solve(max_iter=5)
