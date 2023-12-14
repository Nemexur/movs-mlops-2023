# movs-mlops-2023

## Как подготовить датасет?

0. Сделать `dvc pull` или `mkdir -p data/{gen,cancer}`

### Сгенерированный

1. Генерация

```bash
python scripts/gen_dataset.py --seed 13 --n-samples 150000 > data/gen/full-dataset.jsonl
```

2. Получим train/eval через miller

```bash
mlr --jsonl filter '$part == "train"' + cut -f features,target data/full-dataset.jsonl > data/gen/train.jsonl
mlr --jsonl filter '$part == "eval"' + cut -f features,target data/full-dataset.jsonl > data/gen/eval.jsonl
mlr --jsonl filter '$part == "eval"' + cut -f features data/full-dataset.jsonl > data/gen/eval-no-target.jsonl
```

### Breast Cancer

1. Генерация

```bash
python scripts/cancer_dataset.py --seed 13 > data/cancer/full-dataset.jsonl
```

2. Получим train/eval через miller

```bash
mlr --jsonl filter '$part == "train"' + cut -f features,target data/full-dataset.jsonl > data/cancer/train.jsonl
mlr --jsonl filter '$part == "eval"' + cut -f features,target data/full-dataset.jsonl > data/cancer/eval.jsonl
mlr --jsonl filter '$part == "eval"' + cut -f features data/full-dataset.jsonl > data/cancer/eval-no-target.jsonl
```

## Как обучить модель?

Конфиги для train/infer моделей сделаны через jinja,
поэтому им обязательно нужно добавить переменные: datasets, batch_size, in_features, num_classes, hidden_dim (default=100)

```bash
python experiments/cmd/train.py configs/train.yaml.j2 \
  -d {директория для сохранения checkpoints} \
  --extra-vars datasets={директория с файлами {data/gen,data/cancer}},batch_size={your input},in_features={your input},num_classes={your input}
```

## Как сделать infer модели?

Чтобы все правильно работало и инициализировалось,
значение extra-vars должно быть равно тем, что присутствовали во время запуска train.py.

```bash
python experiments/cmd/infer.py configs/infer.yaml.j2 {директория из пред этапа}/best_iteration/model.safetensors \
  --extra-vars datasets={директория с файлами {data/gen,data/cancer}},batch_size={your input},in_features={your input},num_classes={your input}
```
