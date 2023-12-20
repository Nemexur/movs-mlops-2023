# pyright: reportOptionalMemberAccess=false

from typing import Any, Callable
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from hydra.utils import instantiate
from ignite.engine import EventEnum
from ignite.metrics import Metric
from rich import print_json
import torch

from experiments import settings
from experiments.base import Experiment
from experiments.options import (
    attach_best_exp_saver,
    attach_checkpointer,
    attach_debug_handler,
    attach_log_epoch_metrics,
    attach_metrics,
    attach_progress_bar,
)
from experiments.trainer import Trainer
from experiments.utils import flatten_config


class ClassificationExperiment(Experiment):
    def __init__(
        self,
        exp_config: dict[str, Any] | Callable[[], dict[str, Any]],
        dir: Path | None = None,
        metrics: dict[str, Metric] | None = None,
        trackers_params: dict[str, Any] | None = None,
        events: dict[str, list[tuple[EventEnum, Callable]]] = None,
        seed: int = 13,
        debug: bool = False,
    ) -> None:
        self._config = exp_config if isinstance(exp_config, dict) else exp_config()
        self._dir = dir
        self._seed = seed
        self._debug = debug
        self._metrics = metrics or {}
        self._trackers_params = trackers_params or {}
        self._events = events or {}

    @property
    def metrics(self) -> dict[str, Any]:
        return self._state.metrics

    def run(self) -> Any:
        self._accelerator = self._get_accelerator()
        print_json(data=self._config)
        self._model = self._accelerator.prepare(instantiate(self._config["model"]))
        self._optimizer = self._accelerator.prepare(
            instantiate(self._config["optimizer"])(self._model.parameters())
        )
        max_iters = {k: d.pop("max_iters", None) for k, d in self._config["datasets"].items()}
        self._datasets = {
            key: self._accelerator.prepare_data_loader(
                instantiate(
                    loader,
                    generator=torch.Generator().manual_seed(self._seed),
                    worker_init_fn=seed_worker,
                ),
            )
            for key, loader in self._config["datasets"].items()
        }
        self.trainer = self._get_trainer(self._model, self._optimizer)
        self._state = self.trainer.run(
            self._datasets, max_iters=max_iters, epochs=self._config["epochs"]
        )
        self._accelerator.wait_for_everyone()
        self._accelerator.end_training()

    def interrupt(self) -> None:
        for e in self.trainer.engines.values():
            e.interrupt()

    def clean(self) -> None:
        self._accelerator.free_memory()
        self._accelerator = self.trainer = None
        del self._accelerator, self.trainer

    def _get_accelerator(self) -> Accelerator:
        accelerator = Accelerator(
            log_with=list(self._trackers_params) if len(self._trackers_params) > 0 else None,
            cpu=True,
        )
        if self._dir is not None:
            accelerator.project_configuration = ProjectConfiguration(
                project_dir=str(self._dir),
                automatic_checkpoint_naming=True,
            )
        accelerator.init_trackers(
            project_name=settings.WANDB_PROJECT,
            config=flatten_config(self._config),
            init_kwargs=self._trackers_params,
        )
        self._seed_everything()
        return accelerator

    def _get_trainer(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> Trainer:
        trainer = Trainer(model, optimizer=optimizer, accelerator=self._accelerator)
        if self._debug:
            attach_debug_handler(trainer, num_iters=2000)
        attach_metrics(trainer, self._accelerator, self._metrics)
        attach_progress_bar(
            trainer,
            metric_names={
                "eval": ["loss"] + list(self._metrics),
                "train": ["loss"] + list(self._metrics),
            },
        )
        attach_log_epoch_metrics(trainer, self._accelerator)
        if self._dir is not None:
            attach_checkpointer(
                trainer, self._accelerator, checkpoint_objects=self._metrics.values()
            )
            attach_best_exp_saver(trainer, self._dir, config=self._config)
        for key, e in self._events.items():
            for event, handler in e:
                trainer.add_event(key, event, handler, accelerator=self._accelerator)
        return trainer

    def _seed_everything(self) -> None:
        import os
        import random

        random.seed(self._seed)
        os.environ["PYTHONHASHSEED"] = str(self._seed)
        set_seed(self._seed)


def seed_worker(*_) -> None:
    import random

    import numpy as np

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)  # noqa: NPY002
    random.seed(worker_seed)
