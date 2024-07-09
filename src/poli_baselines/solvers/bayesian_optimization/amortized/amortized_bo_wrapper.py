import numpy as np
from poli.core.abstract_black_box import AbstractBlackBox

from poli_baselines.core.abstract_solver import AbstractSolver
from poli_baselines.solvers.bayesian_optimization.amortized.data import (
    samples_from_arrays,
    Population,
)
from poli_baselines.solvers.bayesian_optimization.amortized.deep_evolution_solver import (
    MutationPredictorSolver,
)
from poli_baselines.solvers.bayesian_optimization.amortized.domains import (
    FixedLengthDiscreteDomain,
    Vocabulary,
)


class AmortizedBOWrapper(AbstractSolver):
    def __init__(self, black_box: AbstractBlackBox, x0: np.ndarray, y0: np.ndarray):
        super().__init__(black_box, x0, y0)
        self.domain = FixedLengthDiscreteDomain(
            vocab=Vocabulary(black_box.get_black_box_info().get_alphabet()),
            length=x0.shape[1],
        )
        self.solver = MutationPredictorSolver(
            domain=self.domain,
            initialize_dataset_fn=lambda *args, **kwargs: self.domain.encode(x0),
        )

    def next_candidate(self) -> np.ndarray:
        samples = samples_from_arrays(
            structures=self.domain.encode(self.x0.tolist()), rewards=self.y0.tolist()
        )
        x = self.solver.propose(num_samples=1, population=Population(samples))
        return np.array([c for c in self.domain.decode(x)[0]])[np.newaxis, :]
