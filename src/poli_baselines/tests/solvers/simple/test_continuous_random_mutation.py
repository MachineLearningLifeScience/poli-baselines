"""This module tests the random search solver"""


def test_continuous_random_mutation_instantiates_and_runs_well():
    from poli import objective_factory

    from poli_baselines.solvers.simple.continuous_random_mutation import \
        ContinuousRandomMutation

    problem = objective_factory.create(
        name="toy_continuous_problem",
        function_name="ackley_function_01",
        n_dimensions=10,
    )
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)

    solver = ContinuousRandomMutation(
        black_box=f,
        x0=x0,
        y0=y0,
    )

    solver.solve(max_iter=5)


if __name__ == "__main__":
    test_continuous_random_mutation_instantiates_and_runs_well()
