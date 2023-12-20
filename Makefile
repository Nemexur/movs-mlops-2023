DIR:=$(shell pwd)


all: help

.PHONY: help
#> Display this message and exit
help:
	@echo "Commands:"
	@awk 'match($$0, "^#>") { sub(/^#>/, "", $$0); doc=$$0; getline; split($$0, c, ":"); cmd=c[1]; print "  \033[00;32m"cmd"\033[0m"":"doc }' ${MAKEFILE_LIST} | column -t -s ":"

.PHONY: install.dep
#> Install dependencies
install.dep:
	poetry install

.PHONY: install.dvc
#> Pull files from DVC
install.dvc:
	poetry run dvc pull

.PHONY: check
#> Run project evaluation jobs
check: install.dep install.dvc
	pre-commit install
	pre-commit run --all-files
	poetry run python experiments/train.py configs/train.yaml.j2 \
		-d movs-mlops-2023-model \
		--wandb \
		--extra-vars "datasets=data/cancer,batch_size=8,in_features=30,num_classes=2"
	poetry run python experiments/infer.py configs/infer.yaml.j2 \
		movs-mlops-2023-model/best_iteration/model.safetensors \
		-o infer-results.csv \
		--extra-vars "datasets=data/cancer,batch_size=8,in_features=30,num_classes=2"
	head infer-results.csv

.PHONY: clean
#> Clean cached files
clean:
	@fd -t d -HI --exclude .venv "__pycache__" --exec rm -rf
