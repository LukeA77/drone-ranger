# Common tasks. Dataset-dependent experiments are listed, in order, in docs/runbook.md.
# On Windows the venv interpreter lives in .venv/Scripts; elsewhere in .venv/bin.
ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python
else
PY ?= .venv/bin/python
endif

.PHONY: setup lint test data-check env midair-eval visdrone-prep \
        demo docs pareto pi-benchmark containers platform-check

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

# Version 2 pipeline latency on a Raspberry Pi 5 (run on the device; needs only the runtime
# dependencies and an ONNX model). FRAMES=<folder of JPEG frames> if the VisDrone validation
# sequence is not on the device; NOTE="..." records anything else running.
pi-benchmark:
	$(PY) scripts/benchmark_v2.py --tag pi5 $(if $(FRAMES),--frames-dir $(FRAMES),) --note "$(NOTE)"

# The runtime image for x86-64 and ARM64 (Raspberry Pi 5), built from docker/Dockerfile.
# Needs Docker with buildx; ARM64 runs under QEMU emulation on an x86 machine.
containers:
	docker buildx build --platform linux/amd64 -f docker/Dockerfile -t aerialrange:amd64 --load .
	docker buildx build --platform linux/arm64 -f docker/Dockerfile -t aerialrange:arm64 --load .

# The whole pipeline in both images on the same frames and model, then the outputs compared
# against the stated tolerances (scripts/platform_check.py). FRAMES=<folder of JPEG frames>,
# for example a VisDrone MOT val sequence; the model is the one configs/run.yaml names.
platform-check: containers
	@test -n "$(FRAMES)" || { echo "FRAMES=<folder of JPEG frames> is required"; exit 1; }
	for a in amd64 arm64; do \
	  docker run --rm --platform linux/$$a -v "$(CURDIR)/weights:/app/weights:ro" \
	    -v "$(abspath $(FRAMES)):/frames:ro" -v "$(CURDIR)/results/edge:/app/results/edge" \
	    aerialrange:$$a python scripts/platform_check.py --tag $$a --frames /frames || exit 1; \
	done
	$(PY) scripts/platform_check.py --compare results/edge/platform_amd64.json results/edge/platform_arm64.json
