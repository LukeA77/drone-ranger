# Common tasks. Dataset-dependent experiments are listed, in order, in docs/runbook.md.
# On Windows the venv interpreter lives in .venv/Scripts; elsewhere in .venv/bin.
ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python
else
PY ?= .venv/bin/python
endif

.PHONY: setup lint test data-check env midair-eval visdrone-prep \
        demo docs pareto

setup:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-lock.txt
	$(PY) -m pip install -e . --no-deps

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

test:
	$(PY) -m pytest --cov=aerialrange --cov-report=term-missing

data-check:
	$(PY) scripts/check_data.py

env:
	$(PY) scripts/env_report.py

# Mid-Air development and calibration work. The test split is NOT run here; see
# docs/runbook.md for the one-off --final commands.
midair-eval:
	$(PY) scripts/make_splits.py
	$(PY) scripts/midair_eval.py --split development
	$(PY) scripts/midair_eval.py --split calibration
	$(PY) scripts/midair_analyse.py --split development
	$(PY) scripts/midair_analyse.py --split calibration

visdrone-prep:
	$(PY) scripts/visdrone_overlap.py
	$(PY) scripts/visdrone_prep.py --mot --det

# Whole pipeline on a synthetic flight, checked against its truth (no datasets needed).
demo:
	$(PY) -m aerialrange.cli demo --out cache/demo

# Rebuild results/headline.json and every document from docs/src/ (fails on any unfilled number).
docs:
	$(PY) scripts/build_docs.py

# Final Pareto chart from the saved benchmark and accuracy records.
pareto:
	$(PY) scripts/pareto.py
