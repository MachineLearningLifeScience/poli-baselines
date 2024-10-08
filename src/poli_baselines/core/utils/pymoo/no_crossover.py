from pymoo.core.crossover import Crossover


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
