import csv
from io import TextIOWrapper
from pathlib import Path
import sys

from accelerate import Accelerator
import click
from hydra.utils import instantiate
from ignite.handlers import EpochOutputStore
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from rich.console import Console
from safetensors.torch import load_model
import torch
import yaml

from experiments.click_options import State, extra_vars_option, name_option, pass_state
from experiments.trainer import Trainer


@click.command(
    help="Run infer.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--out",
    type=click.File("w", encoding="utf-8"),
    help="Output file. By default prints to stdout.",
    default="-",
)
@name_option("exp")
@extra_vars_option
@pass_state
@torch.no_grad()
def main(state: State, config_path: Path, model_path: Path, out: TextIOWrapper) -> None:
    console = Console(file=sys.stderr)
    jinja_env = Environment(
        loader=FileSystemLoader("."), undefined=StrictUndefined, autoescape=True
    )
    config = yaml.safe_load(
        jinja_env.get_template(str(config_path)).render(**(state.extra_vars or {}))
    )
    accelerator = Accelerator()
    console.print_json(data=config)
    model = instantiate(config["model"])
    load_model(model, model_path)
    model, dataset = accelerator.prepare(model, instantiate(config["dataset"], shuffle=False))
    trainer = Trainer(model=model, optimizer=None, accelerator=accelerator)
    EpochOutputStore().attach(trainer.engines["eval"], name="result")
    state = trainer.engines["eval"].run(dataset)
    # Write result
    sample_id = 0
    out_writer = csv.DictWriter(out, fieldnames=("id", "prob", "label"))
    out_writer.writeheader()
    for r in state.result:
        labels = r["logits"].argmax(dim=-1)
        probs = r["probs"][torch.arange(r["probs"].size(0)), labels]
        for p, l in zip(probs.cpu().numpy().tolist(), labels.cpu().numpy().tolist(), strict=True):
            out_writer.writerow({"id": sample_id, "prob": round(p, 4), "label": l})
            sample_id += 1


if __name__ == "__main__":
    main()