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

THIS_DIR = Path(__file__).parent.resolve()
DEFAULT_CONFIG_DIR = THIS_DIR / "hydra_configs"


class Lambo2(AbstractSolver):
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
    ):
        super().__init__(black_box=black_box, x0=x0, y0=y0)
        self.experiment_id = f"{uuid4()}"[:8]
        self.max_epochs_for_retraining = max_epochs_for_retraining
        if config_dir is None:
            config_dir = DEFAULT_CONFIG_DIR

        with hydra.initialize_config_dir(config_dir=str(config_dir)):
            cfg = hydra.compose(config_name=config_name, overrides=overrides)
            OmegaConf.set_struct(cfg, False)

        # Setting the random seed
        if seed is not None:
            cfg.update({"random_seed": seed})
            seed_python_numpy_and_torch(seed)
            L.seed_everything(seed=cfg.random_seed, workers=True)
        self.cfg = cfg

        # TODO: decide on these criteria
        if x0 is None:
            # Run directed evolution or something like that to bootstrap.
            # TODO: implement.
            x0 = ...
        elif x0.shape[0] < cfg.num_samples:
            original_size = x0.shape[0]
            # TODO: implement.
            x0 = ...

        tokenized_x0 = np.array([" ".join(x_i) for x_i in x0])

        if y0 is None:
            y0 = self.black_box(x0)
        elif y0.shape[0] < x0.shape[0]:
            y0 = np.vstack([y0, self.black_box(x0[original_size:])])

        self.history_for_training = {
            "x": [tokenized_x0],
            "y": [y0.flatten()],
        }

        # Pre-training the model.
        MODEL_FOLDER = Path(cfg.data_dir) / self.experiment_id
        MODEL_FOLDER.mkdir(exist_ok=True, parents=True)
        self.model_path = MODEL_FOLDER / "ongoing_model.ckpt"
        self.train_model_with_history(
            save_checkpoint_to=self.model_path,
            max_epochs=cfg.max_epochs,
        )

    def train_model_with_history(
        self,
        load_checkpoint_from: Path | None = None,
        save_checkpoint_to: Path | None = None,
        max_epochs: int = 2,
    ) -> L.LightningModule:
        # TODO: what to do on empty histories? Should
        # we just skip?
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

    def get_candidate_points_from_history(self) -> np.ndarray:
        x = np.concatenate(self.history_for_training["x"], axis=0)
        y = np.concatenate(self.history_for_training["y"], axis=0)
        sorted_y0_idxs = np.argsort(y.flatten())[::-1]
        candidate_points = x[sorted_y0_idxs[: self.cfg.num_samples]]

        return candidate_points

    def step(self):
        """
        Loads the model, runs the optimizer (LaMBO2) for the
        number of steps in the config, computes new proposal,
        evaluates on the black box and updates history.
        """
        # Load the model and optimizer
        model = self.train_model_with_history(
            load_checkpoint_from=self.model_path,
            max_epochs=self.max_epochs_for_retraining,
        )

        # Builds the acquisition function
        candidate_points = self.get_candidate_points_from_history()
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

    def solve(self, max_iter: int = 100) -> None:
        for _ in range(max_iter):
            self.step()
