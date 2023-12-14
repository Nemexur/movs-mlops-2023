from io import TextIOWrapper
import json

import click
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


@click.command(
    help="Generate dataset",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option("--n-samples", type=click.INT, default=100_000, show_default=True)
@click.option("--n-features", type=click.INT, default=50, show_default=True)
@click.option("--n-informative", type=click.INT, default=13, show_default=True)
@click.option("--n-classes", type=click.INT, default=10, show_default=True)
@click.option("--test-size", type=click.FLOAT, default=0.2, show_default=True)
@click.option("--seed", type=click.INT, default=13, show_default=True)
@click.option(
    "-o",
    "--out",
    type=click.File("w", encoding="utf-8"),
    help="Output file. By default prints to stdout.",
    default="-",
)
def main(
    out: TextIOWrapper,
    n_samples: int = 100_000,
    n_features: int = 50,
    n_informative: int = 13,
    n_classes: int = 10,
    test_size: float = 0.2,
    seed: int = 13,
) -> None:
    features, target = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_classes=n_classes,
        random_state=seed,
    )
    train_features, eval_features, train_target, eval_target = train_test_split(
        features, target, test_size=test_size, random_state=seed
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
