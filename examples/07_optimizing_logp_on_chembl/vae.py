from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import kl_divergence
from torch.nn.functional import nll_loss

from torchtext.vocab import Vocab

import pyro
import pyro.distributions as dist
from pyro.distributions import constraints

THIS_DIR = Path(__file__).parent.resolve()


class Encoder(nn.Module):
    """
    Base Encoder Module for VAE.
    Input:
    z_dim -- number of latent dimensions
    hidden_dims -- number of hidden layers
    input_dimt -- size of input NN
    num_categories -- size of input alphabet
    """

    def __init__(
        self, z_dim: int, hidden_dims: int, input_dims: int, num_categories: int
    ):
        super().__init__()
        self.sequence_dims = input_dims
        self.num_classes = num_categories
        encoding_layers = []
        current_dim = input_dims
        for hidden_dim in hidden_dims:
            encoding_layers.append(nn.Linear(current_dim, hidden_dim))
            encoding_layers.append(nn.ReLU(inplace=True))
            current_dim = hidden_dim
        self.encoding_nn = nn.Sequential(*encoding_layers)
        self.mean = nn.Linear(current_dim, z_dim)
        self.log_var = nn.Linear(current_dim, z_dim)

    def forward(self, x):
        x = x.reshape(-1, self.sequence_dims)
        z_loc = self.mean(self.encoding_nn(x))
        z_scale = torch.exp(self.log_var(self.encoding_nn(x)) * 0.5)  # SD from log_VAR
        return z_loc, z_scale


class Decoder(nn.Module):
    """
    Base Decoder Module for VAE.
    Input:
    z_dim -- number of latent dimensions
    hidden_dims -- number of hidden layers
    input_dimt -- size of input NN
    num_categories -- size of input alphabet
    """

    def __init__(
        self,
        z_dim: int,
        hidden_dims: int,
        input_dims: int,
        num_categories: int,
        dropout=0.5,
    ):
        super().__init__()
        decoding_layers = []
        self.categories = num_categories
        self.sequence_length = int(input_dims / num_categories)
        current_dim = z_dim
        for hidden_dim in hidden_dims:
            decoding_layers.append(nn.Linear(current_dim, hidden_dim))
            decoding_layers.append(nn.ReLU(inplace=True))
            decoding_layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        decoding_layers.append(nn.Linear(current_dim, input_dims))
        self.decoding_nn = nn.Sequential(*decoding_layers)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        batch_size = x.shape[0]
        z = self.decoding_nn(x)
        seq_space = z.view(batch_size, self.sequence_length, -1)
        # assert seq_space.shape[2] == self.categories
        loc_img = self.log_softmax(seq_space)
        return loc_img


class VAE(nn.Module):
    """
    Base VAE.
    Input:
    z_dim -- number of latent dimensions
    encoder_dims -- number of encoder layers
    decoder_dims -- number of decoder layers
    input_dims -- size of input
    reference -- reference input
    num_categories -- size of input alphabet
    use_cuda -- utilize CUDA device for computation -- Default=False
    dropout -- dropout fraction -- Default=0.
    """

    def __init__(
        self,
        z_dim: int,
        encoder_dim: int,
        decoder_dim: int,
        input_dims: int,
        reference: np.ndarray,
        num_categories: int,
        device: str = "",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.num_categories = num_categories
        self.sequence_length = int(input_dims / num_categories)
        self.encoder = Encoder(
            z_dim, encoder_dim, input_dims, num_categories=num_categories
        )
        self.decoder = Decoder(
            z_dim,
            decoder_dim,
            input_dims=input_dims,
            num_categories=num_categories,
            dropout=dropout,
        )
        self.mse = nn.MSELoss()
        if device and device.type == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.apply(lambda t: t.to(device))
        elif device and device.type == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.cuda()
        else:
            self.device = torch.device("cpu")
        self.z_dim = z_dim
        self._hidden_dim = encoder_dim, decoder_dim
        self.reference = reference

    def model(self, x, y):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = x.new_zeros(
                torch.Size((x.shape[0], self.z_dim)), device=self.device
            )
            z_scale = x.new_ones(
                torch.Size((x.shape[0], self.z_dim)), device=self.device
            )
            z = pyro.sample(
                "latent", dist.Normal(z_loc, z_scale, constraints.positive).to_event(1)
            )
            loc_seq = self.decoder.forward(z).exp()
            categorical_x = x.argmax(-1)
            pyro.sample(
                "obs",
                dist.Categorical(loc_seq, validate_args=True).to_event(1),
                obs=categorical_x,
            )

    def guide(self, x, y):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder.forward(x)
            pyro.sample(
                "latent", dist.Normal(z_loc, z_scale, constraints.positive).to_event(1)
            )

    def representation(self, z: dist) -> torch.Tensor:
        z_repr = self.decoder(z.loc).exp()
        sample = dist.Categorical(z_repr).sample()
        return sample

    def reconstruct(self, x):
        z_loc, z_scale = self.encoder(x)
        z_dist = dist.Normal(z_loc, z_scale)
        reconstruction = self.representation(z_dist)
        return reconstruction

    def log_p(self, x):
        # TODO: refactor to obtain batched log_p
        z_loc, z_scale = self.encoder(x)
        z_dist = dist.Normal(z_loc, z_scale)
        kld = self.kld_loss(z_dist)
        reconstruction = self.decoder(
            z_dist.loc
        )  # TODO: tell richard that the reconstructions should be a sample instead of a mean
        # nll loss input requires: (batch, categories, data)
        log_p = (
            nll_loss(
                reconstruction.permute(0, 2, 1),
                x.view(self.sequence_length, self.num_categories).argmax(-1)[
                    np.newaxis, :
                ],
                reduction="none",
            )
            .mul(-1)
            .sum(1)
        )
        # log_p = dist.Categorical(self.decoder(z_dist.loc).exp()).log_prob(x.argmax(-1)).sum(1)
        elbo = log_p + kld
        return elbo, log_p, kld

    @staticmethod
    def kld_loss(z_dist: dist):
        prior = dist.Normal(torch.zeros_like(z_dist.loc), torch.ones_like(z_dist.scale))
        kld = kl_divergence(z_dist, prior).sum(dim=1)
        return kld

    def mse_loss(self, x):
        x_construct = self.reconstruct(x).argmax(-1).to(torch.float)
        return self.mse(x_construct, x)

    def mse_diff(self, x, y=None):
        """MSE loss is unintuitive/uninformative in a classification task"""
        if y is None:
            y = self.reference
        x_construct = self.reconstruct(x).argmax(-1).to(torch.float)
        y_construct = self.reconstruct(y).argmax(-1).to(torch.float)
        return self.mse(x_construct, y_construct)

    def latent_sample(self, x, n=10):
        z_loc, z_scale = self.encoder(x)
        return dist.Normal(z_loc, z_scale).rsample([n])


def load_vocab() -> Vocab:
    vocab = torch.load(THIS_DIR / "vocab_chembl.pth")
    return vocab


def load_vae() -> VAE:
    vocab = load_vocab()
    vae = VAE(
        z_dim=10,
        encoder_dim=[1000, 250],
        decoder_dim=[250, 1000],
        input_dims=(len(vocab) + 2) * 300,
        reference=None,
        num_categories=(len(vocab) + 2),
    )
    vae.load_state_dict(torch.load(THIS_DIR / "VAE_CHEMBL.pt", map_location="cpu"))

    return vae
