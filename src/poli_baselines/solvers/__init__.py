from .simple.random_mutation import RandomMutation
from .simple.continuous_random_mutation import ContinuousRandomMutation

try:
    from .bayesian_optimization.latent_space_bayesian_optimization import (
        LatentSpaceBayesianOptimization,
    )
    from .bayesian_optimization.line_bayesian_optimization import LineBO
    from .bayesian_optimization.vanilla_bayesian_optimization import (
        VanillaBayesianOptimization,
    )
except ImportError:
    pass
