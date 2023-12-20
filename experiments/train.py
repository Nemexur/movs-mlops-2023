from pathlib import Path

import click
from hydra.utils import instantiate
from jinja2 import Environment, FileSystemLoader, StrictUndefined
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
    pass_state,
    seed_option,
    wandb_option,
)


@click.command(
    help="Run train.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@name_option("exp")
@dir_option
@seed_option
@debug_option
@wandb_option
@extra_vars_option
@pass_state
def main(state: State, config_path: Path) -> None:
    jinja_env = Environment(
        loader=FileSystemLoader("."), undefined=StrictUndefined, autoescape=True
    )
    config = yaml.safe_load(
        jinja_env.get_template(str(config_path)).render(**(state.extra_vars or {}))
    )
    if state.exp_dir is not None:
        state.exp_dir.mkdir(exist_ok=True)
    exp: Experiment = instantiate(
        config.pop("experiment"),
        exp_config=lambda: config,
        dir=state.exp_dir,
        debug=state.debug,
        seed=state.seed,
        trackers_params=({"wandb": {"name": state.exp_name}} if state.use_wandb else {}),
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
