import numpy as np

from pymoo.core.crossover import Crossover

from poli_baselines.core.utils.pymoo.interface import (
    _from_array_to_dict,
    _from_dict_to_array,
)


class NoCrossover(Crossover):
    """
    This is a dummy crossover that does nothing.
    """

    def __init__(self):
        super().__init__(2, 2)

    def _do(self, _, X, **kwargs):
        return X

    # def _next(self):
    #     return False
