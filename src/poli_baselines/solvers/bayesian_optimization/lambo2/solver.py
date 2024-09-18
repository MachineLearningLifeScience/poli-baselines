from __future__ import annotations
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.util.seeding import seed_python_numpy_and_torch
from poli_baselines.core.abstract_solver import AbstractSolver

CACHE_FOLDER = Path.home() / ".cache" / "poli_baselines" / "lambo2"
CACHE_FOLDER.mkdir(exist_ok=True, parents=True)


class Lambo2(AbstractSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray | None = None,
        y0: np.ndarray | None = None,
        config_path: Path | str = None,
        config_name: str | None = None,
        overrides: list[str] | None = None,
        seed: int | None = None,
    ):
        super().__init__(black_box=black_box, x0=x0, y0=y0)
        with hydra.initialize(config_path=config_path):
            cfg = hydra.compose(config_name=config_name, overrides=overrides)
            OmegaConf.set_struct(cfg, False)

        # set random seed
        if seed is not None:
            cfg.update({"seed": seed})
            seed_python_numpy_and_torch(seed)

        L.seed_everything(seed=cfg.random_seed, workers=True)

        cfg.tasks.protein_generation.protein_seq.data_module.dataset_config = (
            DictConfig({"_target_": "cortex.data.dataset.NumpyDataset", "train": "???"})
        )
        self.cfg = cfg
        tokenized_x0 = np.array([" ".join(x_i) for x_i in x0])
        self.history_for_training = {
            "x": [tokenized_x0],
            "y": [y0.flatten()],
        }

        # TODO: add an ID for the run.
        self.model_path = CACHE_FOLDER / "ongoing_model.ckpt"
        model = self.train_model_with_history(
            save_checkpoint_to=self.model_path,
            max_epochs=cfg.max_epochs,
        )

        if x0 is not None:
            sorted_y0_idxs = np.argsort(y0.flatten())[::-1]
            candidate_points = tokenized_x0[sorted_y0_idxs[:16]]
        else:
            # TODO:
            # What should we do here? Just propose some random
            # sequences?
            candidate_points = ...

        acq_fn_runtime_kwargs = hydra.utils.call(
            cfg.guidance_objective.runtime_kwargs,
            model=model,
            candidate_points=candidate_points,
        )
        acq_fn = hydra.utils.instantiate(
            cfg.guidance_objective.static_kwargs, **acq_fn_runtime_kwargs
        )

        tokenizer_transform = model.root_nodes["protein_seq"].eval_transform
        tokenizer = tokenizer_transform[0].tokenizer
        tok_idxs = tokenizer_transform(candidate_points)
        is_mutable = tokenizer.get_corruptible_mask(tok_idxs)
        tok_idxs = tokenizer_transform(candidate_points)

        self.optimizer = hydra.utils.instantiate(
            cfg.optim,
            params=tok_idxs,
            is_mutable=is_mutable,
            model=model,
            objective=acq_fn,
            constraint_fn=None,
        )

    def train_model_with_history(
        self,
        load_checkpoint_from: Path | None = None,
        save_checkpoint_to: Path | None = None,
        max_epochs: int = 2,
    ) -> L.LightningModule:
        model = hydra.utils.instantiate(self.cfg.tree)
        model.build_tree(self.cfg, skip_task_setup=True)

        # TODO: add loading the previous weights
        if load_checkpoint_from is not None and load_checkpoint_from.exists():
            model.load_state_dict(
                torch.load(load_checkpoint_from, map_location="cpu")["state_dict"]
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
        self.cfg.update({"max_epochs": max_epochs})
        self.trainer: L.Trainer = hydra.utils.instantiate(self.cfg.trainer)
        self.trainer.fit(
            model,
            train_dataloaders=model.get_dataloader(split="train"),
            val_dataloaders=model.get_dataloader(split="val"),
        )

        if save_checkpoint_to:
            self.trainer.save_checkpoint(save_checkpoint_to)

        return model

    def solve(
        self,
        max_iter: int = 100,
        n_initial_points: int = 0,
        seed: int | None = None,
    ) -> None:
        for _ in range(max_iter):
            # Load the model and optimizer
            model = self.train_model_with_history(
                load_checkpoint_from=self.model_path, max_epochs=2
            )
            x = np.concatenate(self.history_for_training["x"], axis=0)
            y = np.concatenate(self.history_for_training["y"], axis=0)

            sorted_y0_idxs = np.argsort(y.flatten())[::-1]
            candidate_points = x[sorted_y0_idxs[:16]]

            acq_fn_runtime_kwargs = hydra.utils.call(
                self.cfg.guidance_objective.runtime_kwargs,
                model=model,
                candidate_points=candidate_points,
            )
            acq_fn = hydra.utils.instantiate(
                self.cfg.guidance_objective.static_kwargs, **acq_fn_runtime_kwargs
            )

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

            # Retraining the model
            self.history_for_training["x"].append(new_designs)
            self.history_for_training["y"].append(new_y.flatten())
