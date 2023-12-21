# movs-mlops-2023

## Описание задачи

Решил взять датасет [Breast Cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).
Решаю задачу классификации по определению злокачественных опухолей на основе признаков,
полученных в результате анализа груди.

## Как подготовить датасет?

0. Сделать `dvc pull` или `mkdir -p data/{gen,cancer}`

### Breast Cancer

1. Генерация

```bash
python scripts/cancer_dataset.py --seed 13 > data/cancer/full-dataset.jsonl
```

2. Получим train/eval через miller

```bash
mlr --jsonl filter '$part == "train"' + cut -f features,target data/cancer/full-dataset.jsonl > data/cancer/train.jsonl
mlr --jsonl filter '$part == "eval"' + cut -f features,target data/cancer/full-dataset.jsonl > data/cancer/eval.jsonl
mlr --jsonl filter '$part == "eval"' + cut -f features data/cancer/full-dataset.jsonl > data/cancer/eval-no-target.jsonl
```

## Модель

Модель можно найти в этом [файле](movs_mlops_2023/models/model.py).

### Как обучить модель?

Конфиги для train/infer моделей сделаны через jinja,
Можно переопределить след параметры: mlflow_uri, epochs, datasets, batch_size, in_features, num_classes, hidden_dim.
Для всех из них в конфиге стоят дефолты.

```bash
python train.py configs/train.yaml.j2 \
  -d {директория для сохранения checkpoints} \
  --extra-vars datasets={директория с файлами {data/gen,data/cancer}},batch_size={your input},in_features={your input},num_classes={your input}
```

## Как сделать infer модели?

Чтобы все правильно работало и инициализировалось,
значение extra-vars должно быть равно тем, что присутствовали во время запуска train.py.

```bash
python infer.py configs/infer.yaml.j2 {директория из пред этапа}/best_iteration/model.safetensors \
  --extra-vars datasets={директория с файлами {data/gen,data/cancer}},batch_size={your input},in_features={your input},num_classes={your input}
```
