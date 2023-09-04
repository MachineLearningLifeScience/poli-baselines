from poli import objective_factory
from poli.core.registry import get_problems
from poli_baselines.solvers.simple.random_mutation import RandomMutation

if __name__ == "__main__":
    assert "our_aloha" in get_problems()

    # Creating an instance of the problem
    problem_info, f, x0, y0, run_info = objective_factory.create(
        name="our_aloha", caller_info=None, observer=None
    )

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
