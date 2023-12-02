# movs-mlops-2023

## Как сделать датасет?

1. Генерация

```bash
poetry run python cmd/gen_dataset.py --seed 13 --n-samples 150000 > data/full-dataset.jsonl
```

2. Получим train/eval через miller

```bash
mlr --jsonl filter '$part == "train"' + cut -f features,target data/full-dataset.jsonl > data/train.jsonl
mlr --jsonl filter '$part == "eval"' + cut -f features,target data/full-dataset.jsonl > data/eval.jsonl
mlr --jsonl filter '$part == "eval"' + cut -f features data/full-dataset.jsonl > data/eval-no-target.jsonl
```
