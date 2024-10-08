import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from poli.core.black_box_information import BlackBoxInformation
from poli.core.util.abstract_observer import AbstractObserver


class SimpleObserverForMads(AbstractObserver):
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

    def save_history(self, path: str) -> None:
        arr_x = []
        for x in self.x_s:
            arr_x.append(np.array(["".join(x_i) for x_i in x]))
        x_s = np.concatenate(arr_x)
        y_s = np.vstack(self.y_s)
        np.savez(path, x_s=x_s, y_s=y_s)

    def save_df_for_mads(self, path: str, batch_size: int) -> None:
        rows = []
        for i, (x, y) in enumerate(zip(self.x_s[1:], self.y_s[1:])):
            batch_id = (i // batch_size) + 1
            for x_i, y_i in zip(x, y.flatten(), strict=True):
                rows.append(
                    {
                        "batch_id": batch_id,
                        "x": "".join(x_i),
                        "y": y_i,
                    }
                )

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)


def plot_best_y(obs: SimpleObserverForMads, ax: plt.Axes, start_from: int = 0):
    best_y = np.maximum.accumulate(np.vstack(obs.y_s).flatten())
    ax.plot(best_y.flatten()[start_from:])
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Best value found")
