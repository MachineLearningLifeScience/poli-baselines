from pymoo.core.mutation import Mutation


class NoMutation(Mutation):
    def __init__(self, prob=1, prob_var=None, **kwargs) -> None:
        super().__init__(prob, prob_var, **kwargs)

    def _do(self, problem, X, **kwargs):
        return X


class WildtypeMutation(Mutation):
    def __init__(self, num_mutations: int = 1, prob=1, prob_var=None, **kwargs) -> None:
        self.num_mutations = num_mutations
        super().__init__(prob, prob_var, **kwargs)

    def _do(self, problem, X, **kwargs):
        return super()._do(problem, X, **kwargs)
