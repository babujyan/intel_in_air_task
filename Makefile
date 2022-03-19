PROJECT = intel_in_air
PYTHON_VERSION=3.8
venv_name = py${PYTHON_VERSION}-${PROJECT}
venv = .venv/${venv_name}

# Commands that activate and run virtual environment versions.
_python = . ${venv}/bin/activate; python
_pip = . ${venv}/bin/activate; pip

default: update_venv submodules
.PHONY: default

pull:
	git pull origin master
	git submodule update
.PHONY: pul

submodules:
	git submodule update --init --recursive
.PHONY: submodules

update_submodules:
	git submodule update --remote
.PHONY: update_submodules

create_venv: ${venv}
.PHONY: create_venv

${venv}: PYTHON_PREFIX=python${PYTHON_VERSION}
${venv}: requirements.txt
	${PYTHON_PREFIX} -m venv ${venv}
	${_pip} install --upgrade pip --cache .tmp/
	${_pip} install -r requirements.txt --cache .tmp/
	export PYTHONPATH="${PYTHONPATH}:."


update_venv: requirements.txt ${venv}
	${_pip} install --upgrade -r requirements.txt --cache .tmp/
	@rm -f .venv/current
	@ln -s ${venv_name} .venv/current
	@echo Success, to activate the development environment, run:
	@echo "\tsource .venv/current/bin/activate"
.PHONY: update_venv


test: LOGGING_LEVEL=ERROR
test:
	PYTHONPATH=. LOGGING_LEVEL=INFO python -m pytest --doctest-modules --durations=10 test  -W 'ignore::DeprecationWarning'
.PHONY: test
