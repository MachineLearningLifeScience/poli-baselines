"""Example: NSGA-2 on discrete inputs using poli baselines."""
import numpy as np

from poli.objective_repository import AlohaProblemFactory
from poli.core.multi_objective_black_box import MultiObjectiveBlackBox

from poli_baselines.solvers import DiscreteNSGAII


if __name__ == "__main__":
    population_size = 15
    batch_size = 10
    max_iterations = 100

    problem_factory = AlohaProblemFactory()
    f_aloha, x0, y0 = problem_factory.create()

    f = MultiObjectiveBlackBox(
        info=f_aloha.info,
        objective_functions=[f_aloha, f_aloha],
    )
    x0 = np.random.choice(f.info.alphabet, size=(batch_size, 5))

    y0 = f(x0)

    solver = DiscreteNSGAII(
        black_box=f,
        x0=x0,
        y0=y0,
        population_size=population_size,
    )

    # One way to run the solver: step by step.
    for _ in range(max_iterations):
        population, fitnesses = solver.step()

        print(population)
        print(fitnesses)

    # Another way to run the solver: all at once.
    # solver.solve(max_iter=max_iterations, verbose=True)
