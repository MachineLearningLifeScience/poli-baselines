"""
In this example, we optimize the QED of molecules
as outputted by a VAE. The VAE is trained on a small
subset of PubChem, and the QED is calculated using poli.
"""
from pathlib import Path
import logging

import numpy as np
import torch
import torch.nn as nn

from poli import objective_factory
from poli_baselines.solvers import LatentSpaceBayesianOptimization

from vae import VAESelfies

THIS_DIR = Path(__file__).parent.resolve()

# TODO: Download the VAE if it doesn't exist


def encode(x: np.ndarray, vae: VAESelfies) -> np.ndarray:
    """
    Encodes a given tensor of int tokens to latent space.
    """
    # Transforming the x to one-hot
    x = torch.from_numpy(x).long()
    x_one_hot = nn.functional.one_hot(x, num_classes=len(vae.tokens_dict)).float()

    x_one_hot = x_one_hot.to(vae.device)
    z = vae.encode(x_one_hot).mean.detach().cpu().numpy()

    return z


def decode(z: np.ndarray, vae: VAESelfies) -> np.ndarray:
    """
    Decodes a given latent code to a tensor of int tokens.
    """
    categorical_dist = vae.decode(torch.from_numpy(z))
    x = categorical_dist.probs.argmax(dim=-1).detach().cpu().numpy()

    return x


if __name__ == "__main__":
    # Loading the VAE
    vae = VAESelfies(latent_dim=2)
    vae.load_state_dict(
        torch.load(
            THIS_DIR / "VAESelfies_TINY-CID-SELFIES-20_latent_dim_2.pt",
            map_location="cpu",
        )
    )

    # Loading the objective
    _, f_qed, _, _, _ = objective_factory.create(
        name="rdkit_qed",
        path_to_alphabet=THIS_DIR / "tokens_TINY-CID-SELFIES-20.json",
        string_representation="SELFIES",
    )
    x_0 = np.array([20 * [1]])
    y_0 = f_qed(x_0)

    # Defining the solver
    solver = LatentSpaceBayesianOptimization(
        black_box=f_qed,
        x0=x_0,
        y0=y_0,
        encoder=lambda x: encode(x, vae),
        decoder=lambda z: decode(z, vae),
    )

    solver.solve(max_iter=200, verbose=True)
    print(solver.history)
