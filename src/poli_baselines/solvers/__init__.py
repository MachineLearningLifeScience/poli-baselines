from .simple.random_mutation import RandomMutation

try:
    from .bayesian_optimization.latent_space_bayesian_optimization import (
        LatentSpaceBayesianOptimization,
    )
    from .bayesian_optimization.line_bayesian_optimization import LineBO
except ImportError:
    pass
