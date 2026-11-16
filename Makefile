.PHONY: install install-dl install-eda lock lint format test data eda baseline compare

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
	$(PY) -m pip install -e ".[base,dl,api,app]"

install-eda:  ## tokenizer + jupyter para notebooks/exploration.ipynb
	$(PY) -m pip install -e ".[base,eda]"

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

data:  ## baixa o B2W (idempotente) e gera data/processed/{train,val,test}.parquet + stats.json
	$(PY) -m src.preprocessing --config configs/data.yaml

# Silencia avisos benignos que poluiriam os outputs versionados do notebook.
eda: export TRANSFORMERS_VERBOSITY := error
eda: export HF_HUB_VERBOSITY := error
eda: export HF_HUB_DISABLE_SYMLINKS_WARNING := 1
eda: export PYTHONWARNINGS := ignore:IProgress not found
eda:  ## executa o notebook de EDA no lugar (precisa de make install-eda e make data)
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/exploration.ipynb

baseline:
	@echo "make baseline: ainda não implementado (Fase 2 — python -m src.baseline --config configs/baseline.yaml)" && exit 1

compare:
	@echo "make compare: ainda não implementado (Fase 6 — gera results/comparison.md)" && exit 1
