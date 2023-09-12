import numpy as np

from pymoo.core.population import Population
from pymoo.core.infill import InfillCriterion


class WildtypeMating(InfillCriterion):
    def __init__(
        self, repair=None, eliminate_duplicates=None, n_max_iterations=100, **kwargs
    ):
        super().__init__(repair, eliminate_duplicates, n_max_iterations, **kwargs)

    def _do(self, problem, pop, n_offsprings, **kwargs):
        off = Population.new(X=pop.get("X")[:n_offsprings])
        for child in off:
            # TODO: Check for padding and mutate until there.
            random_position = np.random.randint(0, len(child.X))
            random_key = f"x_{random_position}"

            mutant_at_position = problem.vars[random_key]._sample(1)[0]
            while mutant_at_position == child.X[random_key]:
                mutant_at_position = problem.vars[random_key]._sample(1)[0]

            child.X[random_key] = mutant_at_position
            child.set("X", child.X)
            # child.set("F", problem.evaluate([child.X], return_values_of=["F"]))

        return off
