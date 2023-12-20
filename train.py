from pathlib import Path

import click
from hydra.utils import instantiate
from jinja2 import Environment, FileSystemLoader, StrictUndefined
import mlflow
from rich import print_json
import torch
import yaml

from experiments.base import Experiment
from experiments.click_options import (
    State,
    debug_option,
    dir_option,
    extra_vars_option,
    name_option,
    no_mlflow_option,
    pass_state,
    seed_option,
)


@click.command(
    help="Run train.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--config-path",
    help="Config path.",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path.cwd() / "configs/train.yaml.j2",
    show_default=True,
)
@name_option("exp")
@dir_option("my-model")
@seed_option
@debug_option
@no_mlflow_option
@extra_vars_option
@pass_state
def main(state: State, config_path: Path) -> None:
    jinja_env = Environment(
        loader=FileSystemLoader([".", "/"]), undefined=StrictUndefined, autoescape=True
    )
    config = yaml.safe_load(
        jinja_env.get_template(str(config_path)).render(**(state.extra_vars or {}))
    )
    if state.exp_dir is not None:
        state.exp_dir.mkdir(exist_ok=True)
    if state.use_mlflow:
        mlflow.set_tracking_uri(uri=config["mlflow_uri"])
    exp: Experiment = instantiate(
        config.pop("experiment"),
        exp_config=lambda: config,
        dir=state.exp_dir,
        debug=state.debug,
        seed=state.seed,
        trackers_params=({"mlflow": {"run_name": state.exp_name}} if state.use_mlflow else {}),
    )
    _ = exp.run()
    print_json(
        data={
            metric: value.item() if isinstance(value, torch.Tensor) else value
            for metric, value in exp.metrics.items()
            if not metric.startswith("_")
        }
    )


if __name__ == "__main__":
    main()
