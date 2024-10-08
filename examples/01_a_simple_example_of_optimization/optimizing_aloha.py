from poli import objective_factory

from poli_baselines.solvers.simple.random_mutation import RandomMutation

if __name__ == "__main__":
    # Creating an instance of the problem
    problem = objective_factory.create(name="aloha")
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)

    # Creating an instance of the solver
    solver = RandomMutation(
        black_box=f,
        x0=x0,
        y0=y0,
    )

    # Running the optimization for 1000 steps,
    # breaking if we find a performance above or
    # equal to 5.0, and printing a small summary
    # at each step.
    solver.solve(max_iter=1000, break_at_performance=5.0, verbose=True)
    print(solver.get_best_solution())

    f.terminate()
