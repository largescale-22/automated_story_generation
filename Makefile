clean: clean-pyc clean-test

##### basic #####
set-git:
	git config --local commit.template .gitmessage
	git config --global core.editor "vim"

set-dev:
	pip3 install -r requirements-dev.txt -r requirements.txt

test:
	pytest tests/

style:
	black --config pyproject.toml .
	isort --settings-path pyproject.toml .
	flake8 .

quality:
	black --config pyproject.toml --check .
	isort --settings-path pyproject.toml --check-only .
	flake8 .

#####  clean  #####
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f .coverage.*
	rm -rf .pytest_cache
	rm -rf .mypy_cache
