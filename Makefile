.PHONY: install run test lint format

install:
	python -m pip install -e ".[dev]"

run:
	python -m uvicorn app.main:app --host 0.0.0.0 --port 8080

test:
	python -m pytest

lint:
	python -m ruff check app tests
	python -m mypy app

format:
	python -m ruff format app tests

