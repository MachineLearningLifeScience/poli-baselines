# Optimizing QED in the latent space of a VAE

This is a toy example of how to optimize the quantitative estimate of druglikeness of small molecules using a Variational Autoencoder. The goal is to map the sequences to latent space, and to run Bayesian Optimization there.

## Creating the objective function

The objective function un mind is `rdkit_qed`. It takes a `path_to_alphabet` and, optionally, a `string_representation` (which should be either `SMILES` or `SELFIES`).

```python
from pathlib import Path
import numpy as np

from poli import objective_factory

THIS_DIR = Path(__file__).parent.resolve()

# Loading the objective
_, f_qed, _, _, _ = objective_factory.create(
    name="rdkit_qed",
    path_to_alphabet=THIS_DIR / "tokens_TINY-CID-SELFIES-20.json",
    string_representation="SELFIES",
)
```

In this case, let's start with the selfies associated to 20 carbons sticked together:
```python
x_0 = np.array([20 * [1]])  # Carbons have index 1 in our alphabet
y_0 = f_qed(x_0)  # ...
```

## About the VAE

The `VAE` we plan to use transforms inputs of fixed length `L=20`. It has `encode: torch.Tensor -> Normal` and `decode: torch.Tensor -> Categorical` methods for encoding and decoding.

[In this link, you will find the weights for a 2-dimensional latent space](https://drive.google.com/file/d/1L49gFMn11Q0e8qIIPoe_m_YjDm0Gtggd/view?usp=drive_link). **Download it and place it in the same folder as this readme**. We can load the VAE as follows:

```python
# Loading the VAE
import torch

from vae import VAESelfies

vae = VAESelfies(latent_dim=2)
vae.load_state_dict(
    torch.load(
        THIS_DIR / "VAESelfies_TINY-CID-SELFIES-20_latent_dim_2.pt",
        map_location="cpu",
    )
)
```

## Defining the solver

We use the solver `LatentSpaceBayesianOptimization` which, besides the black box and initial points, takes a `encoder: np.ndarray -> np.ndarray` and `decoder: np.ndarray -> np.ndarray` functions.
- the `encoder` takes the integer representation of the data, and encodes it to latent space.
- the `decoder` takes the latent codes, and returns the integer representation.

Since we have a `vae` that returns distributions, we need to write small wrappers around `vae.encode` and `vae.decode` to abide by the solver:

```python
import torch.nn as nn

def encode(x: np.ndarray, vae: VAESelfies) -> np.ndarray:
    """
    Encodes a given tensor of int tokens to latent space.
    """
    # Transforming the x to one-hot
    x = torch.from_numpy(x).long()
    x_one_hot = nn.functional.one_hot(x, num_classes=len(vae.tokens_dict)).float()

    x_one_hot = x_one_hot.to(vae.device)
    z = vae.encode(x_one_hot).mean.numpy(force=True)

    return z


def decode(z: np.ndarray, vae: VAESelfies) -> np.ndarray:
    """
    Decodes a given latent code to a tensor of int tokens.
    """
    categorical_dist = vae.decode(torch.from_numpy(z))
    x = categorical_dist.probs.argmax(dim=-1).numpy(force=True)

    return x
```

Now we can create an instance of the solver:

```python
from poli_baselines.solvers import LatentSpaceBayesianOptimization

# Defining the solver
solver = LatentSpaceBayesianOptimization(
    black_box=f_qed,
    x0=x_0,
    y0=y_0,
    encoder=lambda x: encode(x, vae),
    decoder=lambda z: decode(z, vae),
)
```

## Optimizing

Running iterations from this latent space optimization is almost trivial:

```python
solver.solve(max_iter=200, verbose=True)
print(solver.get_best_performance())
```
