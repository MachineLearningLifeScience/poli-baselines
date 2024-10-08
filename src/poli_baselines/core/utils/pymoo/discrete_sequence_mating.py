import numpy as np
from pymoo.core.infill import InfillCriterion
from pymoo.core.population import Population


def _mutate(problem, x_as_dict, num_mutations):
    new_x_as_dict = x_as_dict.copy()
    for _ in range(num_mutations):
        # Select a random position to mutate
        random_position = np.random.randint(0, len(x_as_dict))
        random_key = f"x_{random_position}"

        # Make sure that the mutation is not on padding
        while x_as_dict[random_key] == "":
            random_position = np.random.randint(0, len(x_as_dict))
            random_key = f"x_{random_position}"

        # Mutate the child
        mutant_at_position = problem.vars[random_key]._sample(1)[0]

        # we need to make sure we don't mutate to the same value
        while mutant_at_position == x_as_dict[random_key]:
            mutant_at_position = problem.vars[random_key]._sample(1)[0]

        new_x_as_dict[random_key] = mutant_at_position

    return new_x_as_dict


class DiscreteSequenceMating(InfillCriterion):
    """
    Class for performing discrete sequence mating.

    This discrete sequence mating copies the population and mutates each
    offspring at a random position self.num_mutations times.

    Parameters
    ----------
    num_mutations : int, optional
        The number of mutations to perform on each offspring. Default is 1.
    repair : callable, optional
        A function used to repair the mutated offspring. Default is None.
    eliminate_duplicates : bool, optional
        Whether to eliminate duplicates. Default is True.
    n_max_iterations : int, optional
        The maximum number of iterations. Default is 100.
    **kwargs
        Additional keyword arguments.

    Methods
    -------
    _do(problem, pop, n_offsprings, **kwargs)
        Mate members of a population to produce offsprings.

    """

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
        """Mate members of a population to produce offsprings.

        In this discrete sequence mating, we copy the population and mutate
        each offspring at a random position self.num_mutations times.

        See pymoo's abstract InfillCriterion for details.

        Parameters
        ----------
        problem : pymoo.model.problem.Problem
            The problem to be solved.
        pop : pymoo.core.population.Population
            The population from which to mate.
        n_offsprings : int
            The number of offsprings to produce.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        off : pymoo.core.population.Population
            The offspring population.

        """
        # Copy the population
        pop = Population.new(X=pop.get("X")[:n_offsprings])

        # Mutate the current population
        current_pop = pop.get("X")
        new_pop = []
        for x_as_dict in current_pop:
            mutant_x_as_dict = _mutate(problem, x_as_dict, self.num_mutations)

            while mutant_x_as_dict in new_pop or mutant_x_as_dict in current_pop:
                mutant_x_as_dict = _mutate(problem, x_as_dict, self.num_mutations)

            new_pop.append(mutant_x_as_dict)

        pop = Population.new(X=np.array(new_pop))
        return pop
