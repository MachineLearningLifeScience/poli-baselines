import numpy as np

from pymoo.core.population import Population
from pymoo.core.infill import InfillCriterion


class WildtypeMating(InfillCriterion):
    def __init__(
        self,
        num_mutations: int = 1,
        repair=None,
        eliminate_duplicates=None,
        n_max_iterations=100,
        **kwargs,
    ):
        self.num_mutations = num_mutations
        super().__init__(repair, eliminate_duplicates, n_max_iterations, **kwargs)

    def _do(self, problem, pop, n_offsprings, **kwargs):
        # Copy the population
        off = Population.new(X=pop.get("X")[:n_offsprings])
        for child in off:
            for _ in range(self.num_mutations):
                # Select a random position to mutate
                random_position = np.random.randint(0, len(child.X))
                random_key = f"x_{random_position}"

                # Make sure that the mutation is not on padding
                while child.X[random_key] == "":
                    random_position = np.random.randint(0, len(child.X))
                    random_key = f"x_{random_position}"

                # Mutate the child
                mutant_at_position = problem.vars[random_key]._sample(1)[0]
                while mutant_at_position == child.X[random_key]:
                    mutant_at_position = problem.vars[random_key]._sample(1)[0]
                child.X[random_key] = mutant_at_position

            # Set X.
            child.set("X", child.X)

        return off
