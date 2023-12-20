"""
This module implements SAASBO, a Bayesian optimization algorithm
that leverages certain priors on the kernel hyperparameters to
ensure the selection of sparse axis-aligned subspaces [1].

This implementation is based on BoTorch, and their tutorial
on applying SAASBO [2].

[1] High-Dimensional Bayesian Optimization with Sparse
    Axis-Aligned Subspaces, by Eriksson and Jankowiak, 2021.
    https://arxiv.org/abs/2103.00349
[2] https://botorch.org/tutorials/saasbo
"""
from typing import Type, Tuple
from gpytorch.kernels import Kernel
from gpytorch.means import Mean

import numpy as np
import torch

from botorch import fit_fully_bayesian_model_nuts
from botorch.acquisition import ExpectedImprovement, AcquisitionFunction
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

from poli.core.abstract_black_box import AbstractBlackBox
from poli_baselines.solvers.bayesian_optimization.base_bayesian_optimization.base_bayesian_optimization import (
    BaseBayesianOptimization,
)


class SAASBO(BaseBayesianOptimization):
    """
    SAASBO (Sparse Axis-Aligned Bayesian Optimization) is
    a solver that uses a SAAS prior on the Gaussian
    Process' kernel hyperparameters to encourage
    lengthscales to be sparse and axis-aligned [1].

    This implementation is based on the BoTorch tutorial
    on SAASBO [2].

    Parameters
    ----------
    black_box : AbstractBlackBox
        The black box function to be optimized.
    x0 : np.ndarray
        The initial input samples.
    y0 : np.ndarray
        The corresponding output values of the initial input samples.
    mean : Mean, optional
        The mean function of the Gaussian process, by default None.
    kernel : Kernel, optional
        The kernel function of the Gaussian process, by default None.
    acq_function : type[AcquisitionFunction], optional
        The acquisition function to be used, by default ExpectedImprovement.
    bounds : Tuple[float, float], optional
        The bounds of the input space, by default (-2.0, 2.0).
    penalize_nans_with : float, optional
        The value to penalize NaN values in the acquisition function, by default -10.
    warmup_steps : int, optional
        The number of warm-up steps for the acquisition function, by default 256.
    num_samples : int, optional
        The number of samples when fitting the Gaussian Process using NUTS, by default 128.
    thinning : int, optional
        The thinning factor when fitting the Gaussian Process using NUTS, by default 16.

    Attributes
    ----------
    warmup_steps : int
        The number of warm-up steps for the acquisition function.
    num_samples : int
        The number of samples when fitting the Gaussian Process using NUTS.
    thinning : int
        The thinning factor when fitting the Gaussian Process using NUTS.

    Methods
    -------
    next_candidate()
        Returns the next candidate solution after checking the history. It fits
        a SaasFullyBayesianSingleTaskGP model.

    Notes
    -----
    This implementation is based on BoTorch, and their tutorial

    References
    ----------
    [1] High-Dimensional Bayesian Optimization with Sparse
        Axis-Aligned Subspaces, by Eriksson and Jankowiak, 2021.
        https://arxiv.org/abs/2103.00349
    [2] https://botorch.org/tutorials/saasbo
    """

    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        mean: Mean = None,
        kernel: Kernel = None,
        acq_function: type[AcquisitionFunction] = ExpectedImprovement,
        bounds: Tuple[float, float] = (-2.0, 2.0),
        penalize_nans_with: float = -10,
        warmup_steps: int = 256,
        num_samples: int = 128,
        thinning: int = 16,
    ):
        """
        Initialize the SAASBayesianOptimization solver.

        Parameters
        ----------
        black_box : AbstractBlackBox
            The black box function to be optimized.
        x0 : np.ndarray
            The initial input samples.
        y0 : np.ndarray
            The corresponding output values of the initial input samples.
        mean : Mean, optional
            The mean function of the Gaussian process, by default None.
        kernel : Kernel, optional
            The kernel function of the Gaussian process, by default None.
        acq_function : type[AcquisitionFunction], optional
            The acquisition function to be used, by default ExpectedImprovement.
        bounds : Tuple[float, float], optional
            The bounds of the input space, by default (-2.0, 2.0).
        penalize_nans_with : float, optional
            The value to penalize NaN values in the acquisition function, by default -10.
        warmup_steps : int, optional
            The number of warm-up steps for the acquisition function, by default 256.
        num_samples : int, optional
            The number of samples when fitting the
            Gaussian Process using NUTS, by default 128.
        thinning : int, optional
            The thinning factor when fitting the Gaussian
            Process using NUTS, by default 16.
        """
        super().__init__(
            black_box=black_box,
            x0=x0,
            y0=y0,
            mean=mean,
            kernel=kernel,
            acq_function=acq_function,
            bounds=bounds,
            penalize_nans_with=penalize_nans_with,
        )
        self.warmup_steps = warmup_steps
        self.num_samples = num_samples
        self.thinning = thinning

    def _fit_model(
        self, model: Type[SaasFullyBayesianSingleTaskGP], x: np.ndarray, y: np.ndarray
    ) -> SaasFullyBayesianSingleTaskGP:
        """
        Fits a SAASGP model to the data provided.

        Parameters
        ----------
        model : Type[SaasFullyBayesianSingleTaskGP]
            The model to be fitted.
        x : np.ndarray
            The input data.
        y : np.ndarray
            The output data.

        Returns
        -------
        gp : SaasFullyBayesianSingleTaskGP
            The fitted model.
        """
        gp = model(
            train_X=torch.from_numpy(x).to(torch.get_default_dtype()),
            train_Y=torch.from_numpy(y).to(torch.get_default_dtype()),
        )
        fit_fully_bayesian_model_nuts(
            gp,
            warmup_steps=self.warmup_steps,
            num_samples=self.num_samples,
            thinning=self.thinning,
            disable_progbar=True,
        )
        gp.eval()

        return gp

    def next_candidate(self) -> np.ndarray:
        """
        Returns the next candidate solution
        after checking the history. It fits
        a SaasFullyBayesianSingleTaskGP model.

        Returns
        -------
        candidate: np.ndarray
            The next candidate solution.
        """
        # Build up the history
        x, y = self.get_history_as_arrays()

        # Penalize NaNs
        y[np.isnan(y)] = self.penalize_nans_with

        # Fit a SAASGP
        model = self._fit_model(SaasFullyBayesianSingleTaskGP, x, y)

        # Instantiate the acquisition function
        acq_function = self._instantiate_acquisition_function(model)

        # Optimize the acquisition function
        candidate = self._optimize_acquisition_function(acq_function)

        return candidate
