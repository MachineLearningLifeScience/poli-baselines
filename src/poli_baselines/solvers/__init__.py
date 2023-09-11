from .simple.random_mutation import RandomMutation

try:
    from .bayesian_optimization.latent_space_bayesian_optimization import (
        LatentSpaceBayesianOptimization,
    )
except ImportError:
    pass
