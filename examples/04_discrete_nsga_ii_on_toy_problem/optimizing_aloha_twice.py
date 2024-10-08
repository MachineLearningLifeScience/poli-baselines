"""Example: NSGA-2 on discrete inputs using poli baselines."""

import matplotlib.pyplot as plt
import numpy as np
from poli.core.multi_objective_black_box import MultiObjectiveBlackBox
from poli.objective_repository import AlohaProblemFactory

from poli_baselines.solvers import DiscreteNSGAII

if __name__ == "__main__":
    population_size = 15
    batch_size = 10
    max_iterations = 100

    problem = AlohaProblemFactory().create()
    f_aloha = problem.black_box

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
    _, ax = plt.subplots(1, 1)
    for _ in range(max_iterations):
        ax.clear()
        if len(solver.history["y"]) > 0:
            all_previous_y = np.concatenate(solver.history["y"], axis=0)
            ax.scatter(
                all_previous_y[:, 0], all_previous_y[:, 1], color="gray", alpha=0.5
            )

        population, fitnesses = solver.step()

        print(population)
        print(fitnesses)
        ax.scatter(fitnesses[:, 0], fitnesses[:, 1], color="red")
        plt.pause(0.1)

    # Another way to run the solver: all at once.
    # solver.solve(max_iter=max_iterations, verbose=True)
