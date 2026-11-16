.PHONY: install install-dl lock lint format test data baseline compare

# Usa o Python do .venv se existir (Windows ou Unix); senão, o python do PATH (CI).
ifeq ($(OS),Windows_NT)
VENV_BIN := .venv/Scripts
else
VENV_BIN := .venv/bin
endif
PY := $(if $(wildcard $(VENV_BIN)/python*),$(VENV_BIN)/python,python)

install:  ## base + dev (o mesmo que o CI instala)
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pre_commit install

install-dl:  ## dependências de GPU/LLM (Colab/Kaggle)
	$(PY) -m pip install ".[base,dl,api,app]"

lock:  ## regenera requirements.txt a partir de pyproject.toml (base+dev)
	$(PY) -m uv pip compile pyproject.toml --extra base --extra dev \
		--universal --python-version 3.11 -o requirements.txt

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

test:
	$(PY) -m pytest -m "not slow" --cov=src

data:
	@echo "make data: ainda não implementado (Fase 1 — python -m src.preprocessing)" && exit 1

baseline:
	@echo "make baseline: ainda não implementado (Fase 2 — python -m src.baseline --config configs/baseline.yaml)" && exit 1

compare:
	@echo "make compare: ainda não implementado (Fase 6 — gera results/comparison.md)" && exit 1
