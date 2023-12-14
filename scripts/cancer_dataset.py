from io import TextIOWrapper
import json

import click
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


@click.command(
    help="Breast cancer dataset",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option("--seed", type=click.INT, default=13, show_default=True)
@click.option(
    "-o",
    "--out",
    type=click.File("w", encoding="utf-8"),
    help="Output file. By default prints to stdout.",
    default="-",
)
def main(out: TextIOWrapper, seed: int = 13) -> None:
    dataset = load_breast_cancer()
    features, target = dataset["data"], dataset["target"]
    train_features, eval_features, train_target, eval_target = train_test_split(
        features, target, test_size=0.2, random_state=seed
    )
    for f, t in zip(train_features, train_target, strict=True):
        json.dump(
            {"features": f.astype(np.float32).tolist(), "target": int(t), "part": "train"},
            out,
            ensure_ascii=False,
        )
        out.write("\n")
    for f, t in zip(eval_features, eval_target, strict=True):
        json.dump(
            {"features": f.astype(np.float32).tolist(), "target": int(t), "part": "eval"},
            out,
            ensure_ascii=False,
        )
        out.write("\n")


if __name__ == "__main__":
    main()
