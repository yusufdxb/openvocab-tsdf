PY := uv run python
UV := uv

.PHONY: help sync lint fmt test test-fast info cuda-build clean

help:
	@echo "Targets:"
	@echo "  sync        install/refresh the env"
	@echo "  lint        ruff + black --check"
	@echo "  fmt         ruff --fix + black"
	@echo "  test        pytest (all)"
	@echo "  test-fast   pytest -m 'not slow and not gpu and not dataset'"
	@echo "  info        print env / hardware summary via the CLI"
	@echo "  cuda-build  build the CUDA extension in place"
	@echo "  clean       remove build/.pytest_cache/__pycache__ artifacts"

sync:
	$(UV) sync --extra dev

lint:
	$(UV) run ruff check src tests
	$(UV) run black --check src tests

fmt:
	$(UV) run ruff check --fix src tests
	$(UV) run black src tests

test:
	$(UV) run pytest

test-fast:
	$(UV) run pytest -m "not slow and not gpu and not dataset"

info:
	$(UV) run openvocab-tsdf info --config configs/replica_room0.yaml

cuda-build:
	$(UV) run python setup_cuda.py build_ext --inplace

clean:
	rm -rf build/ dist/ *.egg-info/ src/*.egg-info/
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name .pytest_cache -type d -prune -exec rm -rf {} +
	find . -name .ruff_cache -type d -prune -exec rm -rf {} +
	find . -name .mypy_cache -type d -prune -exec rm -rf {} +
