"""
This module implements LaMBO2 by Gruver, Stanton et al. 2023.

LaMBO2 is an improvement on LaMBO [Stanton et al. 2022], using 
guided discrete diffusion and network ensembles instead of 
latent space optimization using Gaussian Processes.

In this module, we import [`cortex`](https://github.com/prescient-design/cortex)
and use the default configuration files except for the `lambo`
optimizer, which is replaced by a more conservative version.
The exact configuration file can be found alongside this file
in our repository:
https://github.com/MachineLearningLifeScience/poli-baselines/tree/main/src/poli_baselines/solvers/bayesian_optimization/lambo2/hydra_configs

:::{warning}
This optimizer only works for **protein-related** black boxes, like
- `foldx_stability`
- `foldx_sasa`
- `rasp`
- `ehrlich`
:::

"""

from __future__ import annotations
from pathlib import Path
from uuid import uuid4

import hydra
import lightning as L
import numpy as np
import torch
from omegaconf import OmegaConf
from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.util.seeding import seed_python_numpy_and_torch
from poli_baselines.core.abstract_solver import AbstractSolver
from poli_baselines.core.utils.mutations import add_random_mutations_to_reach_pop_size

from IPython import embed

THIS_DIR = Path(__file__).parent.resolve()
DEFAULT_CONFIG_DIR = THIS_DIR / "hydra_configs"


class LaMBO2(AbstractSolver):
    """
    LaMBO2 solver for protein-related black boxes.

    Parameters
    ----------
    black_box : AbstractBlackBox
        The black box to optimize. Must be protein-related. To ensure that the
        black box is protein-related, we verify that the `alphabet` inside the
        `info` attribute of the black box is a protein alphabet.
    x0 : np.ndarray
        The initial solutions to the black box. If not enough solutions are
        provided, the solver will generate random mutants to reach the population
        size specified in the configuration file (as cfg.num_samples).
    y0 : np.ndarray, optional
        The initial evaluations of the black box. If not provided, the solver
        will evaluate the black box on the initial solutions.
    config_dir : Path | str, optional
        The directory where the configuration files are stored. If not provided,
        the default configuration files (stored alongside this file in our
        repository) will be used. If you are interested in modifying the
        configurations, we recommend taking a look at the tutorials inside `cortex`.
    config_name : str, optional
        The name of the configuration file to use. Defaults to "generic_training".
    overrides : list[str], optional
        A list of overrides to apply to the configuration file. For example,
        ["num_samples=10", "max_epochs=5"]. To know what to override, we recommend
        taking a look at the tutorials inside `cortex`.
    seed : int, optional
        The random seed to use. If not provided, we use the seed provided in the
        configuration file. If provided, this seed will override the seed in the
        configuration file.
    max_epochs_for_retraining : int, optional
        The number of epochs to retrain the model after each step. Defaults to 1.
    """

    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray | None = None,
        y0: np.ndarray | None = None,
        config_dir: Path | str | None = None,
        config_name: str = "generic_training",
        overrides: list[str] | None = None,
        seed: int | None = None,
        max_epochs_for_retraining: int = 1,
        restrict_candidate_points_to: np.ndarray | None = None,
    ):
        super().__init__(black_box=black_box, x0=x0, y0=y0)
        self.experiment_id = f"{uuid4()}"[:8]
        self.max_epochs_for_retraining = max_epochs_for_retraining
        self.restrict_candidate_points_to = restrict_candidate_points_to

        if config_dir is None:
            config_dir = DEFAULT_CONFIG_DIR
        with hydra.initialize_config_dir(config_dir=str(config_dir)):
            cfg = hydra.compose(config_name=config_name, overrides=overrides)
            OmegaConf.set_struct(cfg, False)

        # Setting the random seed
        # We are ignoring the seed in the original config file.
        if seed is not None:
            cfg.update({"random_seed": seed})

        seed_python_numpy_and_torch(cfg.random_seed)
        L.seed_everything(seed=cfg.random_seed, workers=True)

        self.cfg = cfg

        if x0 is None:
            raise ValueError(
                "In the Lambo2 optimizer, it is necessary to pass at least "
                "a single solution to the solver through x0."
            )
        elif x0.shape[0] < cfg.num_samples:
            original_size = x0.shape[0]
            x0 = add_random_mutations_to_reach_pop_size(
                x0,
                alphabet=self.black_box.info.alphabet,
                population_size=cfg.num_samples,
            )

        tokenizable_x0 = np.array([" ".join(x_i) for x_i in x0])

        if y0 is None:
            y0 = self.black_box(x0)
        elif y0.shape[0] < x0.shape[0]:
            y0 = np.vstack([y0, self.black_box(x0[original_size:])])

        self.history_for_training = {
            "x": [tokenizable_x0],
            "y": [y0.flatten()],
        }

        # Pre-training the model.
        MODEL_FOLDER = Path(cfg.data_dir) / self.experiment_id
        MODEL_FOLDER.mkdir(exist_ok=True, parents=True)
        self.model_path = MODEL_FOLDER / "ongoing_model.ckpt"
        self._train_model_with_history(
            save_checkpoint_to=self.model_path,
            max_epochs=cfg.max_epochs,
        )

    @property
    def history(self) -> dict[str, list[np.ndarray]]:
        """
        Returns the history of the black box evaluations.

        Returns
        -------
        dict[str, list[np.ndarray]]
            The history of the black box evaluations.
        """
        all_x = np.concatenate(self.history_for_training["x"], axis=0)
        all_y = np.concatenate(self.history_for_training["y"], axis=0)

        return {
            "x": [np.array(["".join(x_i).replace(" ", "")]) for x_i in all_x],
            "y": [np.array([[y_i]]) for y_i in all_y],
        }

    def _train_model_with_history(
        self,
        load_checkpoint_from: Path | None = None,
        save_checkpoint_to: Path | None = None,
        max_epochs: int = 2,
    ) -> L.LightningModule:
        """
        Trains the model with the history of the black box evaluations.

        Parameters
        ----------
        load_checkpoint_from : Path, optional
            The path to the checkpoint to load. If not provided, the model will
            be trained from scratch.
        save_checkpoint_to : Path, optional
            The path to save the checkpoint. If not provided, the model will not
            be saved.
        max_epochs : int, optional
            The number of epochs to train the model. Defaults to 2.
        """
        model = hydra.utils.instantiate(self.cfg.tree)
        model.build_tree(self.cfg, skip_task_setup=True)

        if load_checkpoint_from is not None and load_checkpoint_from.exists():
            model.load_state_dict(
                torch.load(
                    load_checkpoint_from,
                    map_location="cpu",
                )["state_dict"]
            )

        x = np.concatenate(self.history_for_training["x"])
        y = np.concatenate(self.history_for_training["y"])

        task_setup_kwargs = {
            # task_key:
            "generic_task": {
                # dataset kwarg
                "data": {
                    "tokenized_seq": x,
                    "generic_task": y,
                }
            },
            "protein_seq": {
                # dataset kwarg
                "data": {
                    "tokenized_seq": x,
                }
            },
        }

        for task_key, task_obj in model.task_dict.items():
            task_obj.data_module.setup(
                stage="test", dataset_kwargs=task_setup_kwargs[task_key]
            )
            task_obj.data_module.setup(
                stage="fit", dataset_kwargs=task_setup_kwargs[task_key]
            )

        # instantiate trainer, set logger
        self.trainer: L.Trainer = hydra.utils.instantiate(
            self.cfg.trainer, max_epochs=max_epochs
        )
        self.trainer.fit(
            model,
            train_dataloaders=model.get_dataloader(split="train"),
            val_dataloaders=model.get_dataloader(split="val"),
        )

        if save_checkpoint_to:
            self.trainer.save_checkpoint(save_checkpoint_to)

        return model
    
    def get_candidate_points(self):
        if self.restrict_candidate_points_to is not None:
            # TODO: make sure the array is of self.cfg.num_samples size.
            # TODO: THey need to be in the "A A A A" format that lambo expects.
            # Let's assume that the user passes a wildtype as
            # np.array(["AAAAA"]) or np.array(["A", "A", "A", ...]).
            assert len(self.restrict_candidate_points_to.shape) == 1
            tokenizable_candidate_point = " ".join(self.restrict_candidate_points_to)
            candidate_points = np.array([tokenizable_candidate_point for _ in range(self.cfg.num_samples)])

            embed()

            return candidate_points
        else:
            return self.get_candidate_points_from_history()

    def get_candidate_points_from_history(self) -> np.ndarray:
        """
        Returns the current best population (whose size is specified in the
        configuration file as cfg.num_samples) from the history of the black
        box evaluations.
        """
        x = np.concatenate(self.history_for_training["x"], axis=0)
        y = np.concatenate(self.history_for_training["y"], axis=0)
        sorted_y0_idxs = np.argsort(y.flatten())[::-1]
        candidate_points = x[sorted_y0_idxs[: self.cfg.num_samples]]

        return candidate_points

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads the model, runs the optimizer (LaMBO2) for the
        number of steps in the config, computes new proposal,
        evaluates on the black box and updates history.
        """
        # Load the model and optimizer
        model = self._train_model_with_history(
            load_checkpoint_from=self.model_path,
            max_epochs=self.max_epochs_for_retraining,
        )

        # Builds the acquisition function
        candidate_points = self.get_candidate_points()
        acq_fn_runtime_kwargs = hydra.utils.call(
            self.cfg.guidance_objective.runtime_kwargs,
            model=model,
            candidate_points=candidate_points,
        )
        acq_fn = hydra.utils.instantiate(
            self.cfg.guidance_objective.static_kwargs, **acq_fn_runtime_kwargs
        )

        # Builds the optimizer
        tokenizer_transform = model.root_nodes["protein_seq"].eval_transform
        tokenizer = tokenizer_transform[0].tokenizer

        # Making sure the model doesn't edit length
        if not self.cfg.allow_length_change:
            tokenizer.corruption_vocab_excluded.add(
                "-"
            )  # prevent existing gap tokens from being masked
            tokenizer.sampling_vocab_excluded.add(
                "-"
            )  # prevent any gap tokens from being sampled

        tok_idxs = tokenizer_transform(candidate_points)
        is_mutable = tokenizer.get_corruptible_mask(tok_idxs)
        tok_idxs = tokenizer_transform(candidate_points)
        optimizer = hydra.utils.instantiate(
            self.cfg.optim,
            params=tok_idxs,
            is_mutable=is_mutable,
            model=model,
            objective=acq_fn,
            constraint_fn=None,
        )

        # Compute proposals using the optimizer
        for _ in range(self.cfg.num_steps):
            # Take a step on the optimizer, diffusing towards promising sequences.
            optimizer.step()

        # Get the most promising sequences from the optimizer
        best_solutions = optimizer.get_best_solutions()
        new_designs = best_solutions["protein_seq"].values
        new_designs_for_black_box = np.array(
            [seq.replace(" ", "") for seq in new_designs]
        )

        # Evaluate the black box
        new_y = self.black_box(new_designs_for_black_box)
        print(new_y)

        # Updating the history that is used for training.
        self.history_for_training["x"].append(new_designs)
        self.history_for_training["y"].append(new_y.flatten())

        return new_designs_for_black_box, new_y

    def solve(self, max_iter: int = 10) -> None:
        """
        Solves the black box optimization problem for a maximum of `max_iter`
        iterations.

        Parameters
        ----------
        max_iter : int, optional
            The maximum number of iterations to run the solver. Defaults to 10.
        """
        for _ in range(max_iter):
            self.step()
