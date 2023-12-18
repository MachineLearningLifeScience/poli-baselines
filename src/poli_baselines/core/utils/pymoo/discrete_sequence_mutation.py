from pymoo.core.mutation import Mutation


class NoMutation(Mutation):
    """
    Class representing a mutation operator that performs no mutation.

    Parameters
    ----------
    prob : float, optional
        The probability of mutation. Default is 1.
    prob_var : float or None, optional
        The probability variance. Default is None.
    **kwargs
        Additional keyword arguments.

    Methods
    -------
    _do(problem, X, **kwargs)
        Perform the mutation operation.

    Returns
    -------
    numpy.ndarray
        The mutated population.

    """

    def __init__(self, prob=1, prob_var=None, **kwargs) -> None:
        super().__init__(prob, prob_var, **kwargs)

    def _do(self, problem, X, **kwargs):
        return X


class DiscreteSequenceMutation(Mutation):
    """
    Discrete sequence mutation operator for genetic/evolutionary algorithms.

    Parameters
    ----------
    num_mutations : int, optional
        The number of mutations to perform on each individual sequence. Default is 1.
    prob : float, optional
        The probability of performing a mutation on each element of the sequence. Default is 1.
    prob_var : float or None, optional
        The variance of the mutation probability. If None, the probability is fixed. Default is None.
    **kwargs
        Additional keyword arguments. See pymoo's abstract Mutation for details.

    """

    def __init__(self, num_mutations: int = 1, prob=1, prob_var=None, **kwargs) -> None:
        self.num_mutations = num_mutations
        super().__init__(prob, prob_var, **kwargs)

    def _do(self, problem, X, **kwargs):
        return super()._do(problem, X, **kwargs)
