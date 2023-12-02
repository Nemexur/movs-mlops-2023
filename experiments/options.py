# pyright: reportOptionalSubscript=false, reportOptionalMemberAccess=false

from typing import Any, Iterable
from pathlib import Path
import shutil
import tarfile
import tempfile

from accelerate import Accelerator
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import Metric, MetricUsage
from loguru import logger
import torch
import yaml

from experiments.trainer import Trainer

BEST_ITERATION_PATH = "best_iteration"


def attach_metrics(trainer: Trainer, metrics: dict[str, Metric] | None = None) -> None:
    if metrics is None:
        return
    metric_usage = MetricUsage(
        started=Events.EPOCH_STARTED,
        completed=Events.ITERATION_COMPLETED,
        iteration_completed=Events.ITERATION_COMPLETED,
    )
    for m in metrics.values():
        m._output_transform = lambda out: (out["logits"], out["target"])  # noqa: W291, SLF001
    for e in trainer.engines.values():
        e.state_dict_user_keys.append("metrics")
        for m_name, m in metrics.items():
            m.attach(e, name=m_name, usage=metric_usage)


def attach_checkpointer(
    trainer: Trainer,
    accelerator: Accelerator,
    checkpoint_objects: Iterable[object] | None = None,
) -> None:
    def save_handler(engine: Engine) -> None:
        engine.state.save_location = accelerator.save_state()
        if accelerator.is_local_main_process:
            logger.info(f"checkpointer: saved checkpoint in {engine.state.save_location}")

    def save_best_handler(engine: Engine) -> None:
        save_dir = Path(accelerator.project_dir) / BEST_ITERATION_PATH
        if accelerator.is_local_main_process:
            shutil.copytree(engine.state.save_location, save_dir, dirs_exist_ok=True)

    for e in trainer.engines.values():
        accelerator.register_for_checkpointing(e)
    for m in checkpoint_objects or []:
        accelerator.register_for_checkpointing(m)
    trainer.add_event("eval", Events.COMPLETED, save_handler)
    trainer.add_event("eval", Events.COMPLETED, save_best_handler)


def attach_progress_bar(
    trainer: Trainer, metric_names: dict[str, str | list[str]] | None = None
) -> None:
    metric_names = metric_names or {}
    for key, e in trainer.engines.items():
        pbar = ProgressBar(
            persist=True,
            bar_format=(
                "{desc} [{n_fmt}/{total_fmt}] "
                "{percentage:3.0f}%|{bar}|{postfix} "
                "({elapsed}<{remaining}, {rate_fmt})"
            ),
            desc="\033[33m" + key.capitalize() + "\033[00m",
        )
        pbar.attach(e, metric_names=metric_names.get(key))


def attach_debug_handler(trainer: Trainer, num_iters: int = 100) -> None:
    def handler(engine: Engine) -> None:
        if engine.state.epoch_iteration < num_iters:
            return
        engine.terminate_epoch()

    for e in trainer.engines:
        trainer.add_event(e, Events.ITERATION_COMPLETED, handler)


def attach_log_epoch_metrics(trainer: Trainer, accelerator: Accelerator) -> None:
    def handler(engine: Engine) -> None:
        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in engine.state.metrics.items()
            if not k.startswith("_")
        }
        if not accelerator.is_local_main_process:
            return
        if len(metrics) == 0:
            return
        run_type = engine.state.name
        logger.info(run_type.capitalize())
        max_length = max(len(x) for x in metrics)
        for metric in sorted(metrics, key=lambda x: (len(x), x)):
            metric_value = metrics.get(metric)
            if isinstance(metric_value, (float, int)):
                logger.info(f"{metric.ljust(max_length)} | {metric_value:.4f}")
        accelerator.log({f"{k}_epoch/{run_type}": v for k, v in metrics.items()})

    for e in trainer.engines:
        trainer.add_event(e, Events.EPOCH_COMPLETED, handler)


def attach_best_exp_saver(trainer: Trainer, dir: Path, config: dict[str, Any]) -> None:
    def handler() -> None:
        exp_archive = dir / "experiment.tar.gz"
        with tempfile.TemporaryDirectory() as tmpdir, tarfile.open(exp_archive, "w:gz") as archive:
            config_path = Path(tmpdir) / "config.yaml"
            with config_path.open("w", encoding="utf-8") as file:
                yaml.dump(config, file, indent=2)
            archive.add(config_path, arcname=config_path.name)
            archive.add(dir / BEST_ITERATION_PATH, arcname=BEST_ITERATION_PATH)

    if not dir.is_dir():
        logger.error(f"attach_best_exp_saver works with directories only (path={dir})")
        return
    trainer.add_event("train", Events.COMPLETED, handler)
