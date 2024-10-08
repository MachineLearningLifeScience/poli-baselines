"""
This script tests Line Bayesian Optimization [1] on the Ackley function
in several dimensions.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from poli import objective_factory

from poli_baselines.core.utils.visualization.objective_functions import (
    plot_objective_function,
)
from poli_baselines.solvers import LineBO

warnings.filterwarnings("ignore", module="botorch")

if __name__ == "__main__":
    problem = objective_factory.create(
        name="toy_continuous_problem",
        function_name="ackley_function_01",
        n_dimensions=2,
    )
    f_ackley = problem.black_box

    x0 = np.random.randn(2).reshape(1, -1).clip(-2.0, 2.0)
    y0 = f_ackley(x0)

    line_bo = LineBO(
        black_box=f_ackley,
        x0=x0,
        y0=y0,
        type_of_line="random",
    )

    # Usually, one could just run:
    # line_bo.solve(max_iter=10)

    # But since we want to visualize what's happening, we'll run it
    # step by step:
    _, (
        ax_objective_function,
        ax_model_prediction,
        ax_acquisition_in_line,
    ) = plt.subplots(1, 3, figsize=(3 * 5, 5))
    for _ in range(20):
        # At each step, "x = solver.next_candidate()" is called. In
        # the case of BO-related implementations, this step updates
        # the underlying GP model, and maximizes the acquisition
        # function to find the next candidate solution.

        # The GP model can be found under solver.model
        line_bo.step()

        # Plotting the objective
        plot_objective_function(
            f_ackley,
            ax=ax_objective_function,
            limits=line_bo.bounds,
            cmap="jet",
        )

        # Plotting the GP model's predictions
        line_bo.plot_model_predictions(ax=ax_model_prediction)

        # Plotting the acquisition function in the current random line
        line_bo.plot_acquisition_function_in_line(ax=ax_acquisition_in_line)

        # Animating the plot
        ax_objective_function.set_title("Objective function")
        ax_model_prediction.set_title("GP model predictions")
        ax_acquisition_in_line.set_title("Acquisition function in line")
        plt.tight_layout()
        plt.pause(1)

        # Clearing the axes
        ax_objective_function.clear()
        ax_model_prediction.clear()
        ax_acquisition_in_line.clear()
