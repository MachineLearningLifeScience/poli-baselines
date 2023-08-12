from typing import Tuple

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation


class AlohaBlackBox(AbstractBlackBox):
    def __init__(self, L: int = 5):
        super().__init__(L=L)

    # The only method you have to define
    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        matches = x == np.array(["A", "L", "O", "H", "A"])
        return np.sum(matches, axis=1, keepdims=True)


class AlohaProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        alphabet_symbols = ["A", ...]
        alphabet = {symbol: i for i, symbol in enumerate(alphabet_symbols)}

        return ProblemSetupInformation(
            name="aloha",
            max_sequence_length=5,
            aligned=True,
            alphabet=alphabet,
        )

    def create(self, seed: int = 0) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        L = self.get_setup_information().get_max_sequence_length()
        f = AlohaBlackBox(L=L)
        x0 = np.array([["A", "L", "O", "O", "F"]])

        return f, x0, f(x0)


if __name__ == "__main__":
    from poli import objective_factory
    from poli.core.registry import register_problem

    # (once) we have to register our factory
    aloha_problem_factory = AlohaProblemFactory()
    register_problem(
        aloha_problem_factory,
        conda_environment_location="/Users/migd/anaconda3/envs/poli-dev",
    )

    # now we can instantiate our objective
    problem_name = aloha_problem_factory.get_setup_information().get_problem_name()
    problem_info, f, x0, y0, run_info = objective_factory.create(
        problem_name, caller_info=None, observer=None
    )

    print(f(x0[:1, :]))
    f.terminate()
