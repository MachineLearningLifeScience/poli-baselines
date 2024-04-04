import numpy as np
from poli.core.abstract_black_box import AbstractBlackBox


class PoliFunctionWrapper:
    def __init__(self, f: AbstractBlackBox):
        self.f = f

    def __call__(self, *args, **kwargs):
        x = args[0]
        return np.squeeze(self.f(x[np.newaxis, :]))
