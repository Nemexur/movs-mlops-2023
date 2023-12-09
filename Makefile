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

.PHONY: clean
#> Clean cached files
clean:
	@fd -t d -HI --exclude .venv "__pycache__" --exec rm -rf
