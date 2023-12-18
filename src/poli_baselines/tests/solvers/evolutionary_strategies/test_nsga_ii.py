"""This module tests our implementation of NSGA-II."""

import numpy as np


def test_running_nsga_ii():
    from poli_baselines.solvers import Discrete_NSGA_II
    from poli import objective_factory
    from poli.core.multi_objective_black_box import MultiObjectiveBlackBox

    population_size = 10

    _, f_aloha, x0, y0, _ = objective_factory.create(
        name="aloha",
    )
    alphabet = f_aloha.info.alphabet

    # Randomly choosing 10 5-letter arrays
    x0 = np.random.choice(alphabet, size=(population_size, 5))

    f = MultiObjectiveBlackBox(
        info=f_aloha.info,
        objective_functions=[f_aloha, f_aloha],
    )

    y0 = f(x0)

    solver = Discrete_NSGA_II(
        black_box=f,
        x0=x0,
        y0=y0,
        population_size=population_size,
        initialize_with_x0=True,
    )

    solver.solve(max_iter=5, verbose=True)


if __name__ == "__main__":
    test_running_nsga_ii()
