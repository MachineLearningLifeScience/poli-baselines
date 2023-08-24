"""
In this example, we optimize the QED of molecules
as outputted by a VAE. The VAE is trained on a small
subset of PubChem, and the QED is calculated using poli.
"""
from pathlib import Path
from typing import Dict
import warnings

import numpy as np
import torch
import torch.nn as nn

from botorch.exceptions import InputDataWarning

from poli import objective_factory
from poli_baselines.solvers import LatentSpaceBayesianOptimization

from vae import VAE, load_vae, load_vocab

THIS_DIR = Path(__file__).parent.resolve()

# TODO: Download the VAE if it doesn't exist

# Ignoring input data warnings
warnings.filterwarnings("ignore", category=InputDataWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def encode(x: np.ndarray, vae: VAE, alphabet: Dict[str, int]) -> np.ndarray:
    """
    Encodes a given tensor of int tokens to latent space.
    """
    # Transforming the x to one-hot
    x = torch.from_numpy(x).long()
    x_one_hot = nn.functional.one_hot(x, num_classes=len(alphabet)).float()

    x_one_hot = x_one_hot.to(vae.device)
    z, _ = vae.encoder(x_one_hot)

    return z.numpy(force=True)


def decode(z: np.ndarray, vae: VAE) -> np.ndarray:
    """
    Decodes a given latent code to a tensor of int tokens.
    """
    x = vae.decoder(torch.from_numpy(z)).argmax(dim=-1)

    return x.numpy(force=True)


if __name__ == "__main__":
    # Loading the VAE
    SEQUENCE_LENGTH = 300
    vae = load_vae()
    vocab = load_vocab()
    alphabet = {symbol: index for index, symbol in enumerate(vocab.vocab.itos_)}
    alphabet["[unk]"] = 60
    alphabet["[unk2]"] = 61

    # Loading the objective
    _, f_qed, _, _, _ = objective_factory.create(
        name="rdkit_qed",
        alphabet=alphabet,
        string_representation="SELFIES",
    )

    # A single carbon.
    x_0 = np.array([[1] + (SEQUENCE_LENGTH - 1) * [0]])
    y_0 = f_qed(x_0)

    # Defining the solver
    solver = LatentSpaceBayesianOptimization(
        black_box=f_qed,
        x0=x_0,
        y0=y_0,
        encoder=lambda x: encode(x, vae, alphabet),
        decoder=lambda z: decode(z, vae),
    )

    solver.solve(max_iter=150, verbose=True)
    best_molecule_as_ints = solver.get_best_solution()[0]
    inverse_alphabet = {v: k for k, v in f_qed.alphabet.items()}
    best_molecule = "".join(
        [
            inverse_alphabet[idx]
            for idx in best_molecule_as_ints
            if inverse_alphabet[idx] != "[nop]"
        ]
    )
    print(f"Best molecule: {best_molecule}")
    print(f"Best QED: {solver.get_best_performance()}")
