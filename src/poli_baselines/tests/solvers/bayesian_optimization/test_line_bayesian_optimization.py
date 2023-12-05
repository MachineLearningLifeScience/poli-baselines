"""
This module tests line bayesian optimization and its utilities.
"""


def test_line_bayesian_optimization_instantiates_and_runs_well():
    from poli_baselines.solvers import LineBO
    from poli import objective_factory

    _, f, x0, y0, _ = objective_factory.create(
        name="toy_continuous_problem",
        function_name="ackley_function_01",
        n_dimensions=10,
    )

    solver = LineBO(
        black_box=f,
        x0=x0,
        y0=y0,
        type_of_line="random",
    )

    solver.solve(max_iter=3)
