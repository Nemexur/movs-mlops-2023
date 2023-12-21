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
#> Run project evaluation steps
check: install.dep dco.up
	poetry run pre-commit install
	poetry run pre-commit run --all-files
	rm -rf $(DIR)/my-model
	poetry run python train.py --extra-vars "mlflow_uri=http://localhost:8080"
	poetry run python infer.py -o - | head

.PHONY: dco.kill
#> Kill docker-compose services
dco.kill:
	@$(FLAGS) docker-compose rm --stop --force

.PHONY: dco.up
#> Run docker-compose services
dco.up: dco.kill
	@$(FLAGS) docker-compose up --build --force-recreate --remove-orphans -d

.PHONY: clean
#> Clean cached files
clean:
	@fd -t d -HI --exclude .venv "__pycache__" --exec rm -rf
