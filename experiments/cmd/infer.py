from pathlib import Path

from accelerate import Accelerator
import click
from hydra.utils import instantiate
from ignite.handlers import EpochOutputStore
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from rich import print_json
from safetensors.torch import load_model
import yaml

from experiments.click_options import State, extra_vars_option, name_option, pass_state
from experiments.trainer import Trainer


@click.command(
    help="Run infer.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@name_option("exp")
@extra_vars_option
@pass_state
def main(state: State, config_path: Path, model_path: Path) -> None:
    jinja_env = Environment(
        loader=FileSystemLoader("."), undefined=StrictUndefined, autoescape=True
    )
    config = yaml.safe_load(
        jinja_env.get_template(str(config_path)).render(**(state.extra_vars or {}))
    )
    accelerator = Accelerator()
    print_json(data=config)
    model = instantiate(config["model"])
    load_model(model, model_path)
    model = accelerator.prepare(model)
    dataset = accelerator.prepare_data_loader(instantiate(config["dataset"], shuffle=False))
    trainer = Trainer(model=model, optimizer=None, accelerator=accelerator)
    EpochOutputStore().attach(trainer.engines["eval"], name="result")
    state = trainer.engines["eval"].run(dataset)
    print(state)


if __name__ == "__main__":
    main()
