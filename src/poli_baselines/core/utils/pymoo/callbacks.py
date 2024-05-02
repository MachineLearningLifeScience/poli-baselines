from typing import Iterable

import numpy as np

from pymoo.core.callback import Callback

from poli_baselines.core.step_by_step_solver import StepByStepSolver

from .interface import _from_dict_to_array


class SaveHistoryAndCallOtherCallbacks(Callback):
    def __init__(self, solver: StepByStepSolver, callbacks: Iterable[Callback]):
        super().__init__()
        self.solver = solver
        self.callbacks = callbacks

    def notify(self, algorithm):
        x_as_array = np.vstack(
            [_from_dict_to_array(x_i) for x_i in algorithm.pop.get("X")]
        )

        self.solver.update(x_as_array, -algorithm.pop.get("F"))

        if self.callbacks is not None:
            for callback in self.callbacks:
                callback(self.solver)
