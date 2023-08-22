"""
Implements a Variational Autoencoder that can be trained on
SELFIES data. It is originally designed to handle the
TINY-CID-SELFIES-20 datasets, but it could be used on the 
SMALL-CID-SELFIES-20 with a little bit of work.
"""
from typing import Tuple
from pathlib import Path
import json

import torch
import torch.nn as nn

from torch.distributions import Normal, Categorical

THIS_DIR = Path(__file__).parent.resolve()


class VAESelfies(nn.Module):
    def __init__(
        self,
        dataset_name: str = "TINY-CID-SELFIES-20",
        max_token_length: int = 20,
        latent_dim: int = 64,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.max_token_length = max_token_length
        self.latent_dim = latent_dim

        assert max_token_length == int(dataset_name.split("-")[-1]), (
            f"max_token_length ({max_token_length}) must match the "
            f"dataset_name's token size ({dataset_name})!"
        )

        self.device = device

        # Load the token dictionary
        with open(THIS_DIR / f"tokens_{dataset_name}.json", "r") as fp:
            self.tokens_dict = json.load(fp)

        # Define the input length: length of a given SELFIES
        # (always padded to be {max_length}), times the number of tokens
        self.input_length = max_token_length * len(self.tokens_dict)

        # Define the model
        self.encoder = nn.Sequential(
            nn.Linear(self.input_length, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.encoder_mu = nn.Linear(256, latent_dim)
        self.encoder_log_var = nn.Linear(256, latent_dim)

        # The decoder, which outputs the logits of the categorical
        # distribution over the vocabulary.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.input_length),
        )

        # Defines the prior
        self.p_z = Normal(
            loc=torch.zeros(latent_dim, device=device),
            scale=torch.ones(latent_dim, device=device),
        )

        # Moves to device
        self.to(device)

    def encode(self, x: torch.Tensor) -> Normal:
        """
        Computes the approximate posterior q(z|x) over
        the latent variable z.
        """
        hidden = self.encoder(x.flatten(start_dim=1).to(self.device))
        mu = self.encoder_mu(hidden)
        log_var = self.encoder_log_var(hidden)

        return Normal(loc=mu, scale=torch.exp(0.5 * log_var))

    def decode(self, z: torch.Tensor) -> Categorical:
        """
        Returns a categorical likelihood over the vocabulary
        """
        logits = self.decoder(z.to(self.device))

        # The categorical distribution expects (batch_size, ..., num_classes)
        return Categorical(
            logits=logits.reshape(-1, self.max_token_length, len(self.tokens_dict))
        )

    def forward(self, x: torch.Tensor) -> Tuple[Normal, Categorical]:
        """
        Computes a forward pass through the VAE, returning
        the distributions q_z_given_x and p_x_given_z.
        """
        q_z_given_x = self.encode(x)
        z = q_z_given_x.rsample()

        p_x_given_z = self.decode(z)

        return q_z_given_x, p_x_given_z

    def loss_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the ELBO loss for a given batch {x}.
        """
        q_z_given_x, p_x_given_z = self.forward(x)

        # Computes the KL divergence between q(z|x) and p(z)
        kl_div = torch.distributions.kl_divergence(q_z_given_x, self.p_z).sum(dim=-1)

        # Computes the reconstruction loss
        recon_loss = -p_x_given_z.log_prob(x.argmax(dim=-1).to(self.device)).sum(dim=-1)

        # Computes the ELBO loss
        return (kl_div + recon_loss).mean()
