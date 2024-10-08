import matplotlib.pyplot as plt
import numpy as np
from poli.core.black_box_information import BlackBoxInformation
from poli.core.util.abstract_observer import AbstractObserver


class SimpleObserver(AbstractObserver):
    def __init__(self) -> None:
        self.x_s = []
        self.y_s = []
        super().__init__()

    def initialize_observer(
        self, problem_setup_info: BlackBoxInformation, caller_info: object, seed: int
    ) -> object: ...

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        self.x_s.append(x)
        self.y_s.append(y)


def plot_best_y(obs: SimpleObserver, ax: plt.Axes, start_from: int = 0):
    best_y = np.maximum.accumulate(np.vstack(obs.y_s).flatten())
    ax.plot(best_y.flatten()[start_from:])
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Best value found")
