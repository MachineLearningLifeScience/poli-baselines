"""This module tests the random search solver"""


def test_continuous_random_mutation_instantiates_and_runs_well():
    from poli_baselines.solvers import ContinuousRandomMutation
    from poli import objective_factory

    _, f, x0, y0, _ = objective_factory.create(
        name="toy_continuous_problem",
        function_name="ackley_function_01",
        n_dimensions=10,
    )

    solver = ContinuousRandomMutation(
        black_box=f,
        x0=x0,
        y0=y0,
    )

    solver.solve(max_iter=5)