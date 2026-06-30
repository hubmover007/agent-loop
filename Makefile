.PHONY: install test test-quick test-cov lint run clean

install:
	pip install -e ".[dev]"

test:
	python3 -m pytest tests/ -v

test-quick:
	python3 -m pytest tests/ -q

test-cov:
	python3 -m pytest tests/ --cov=src --cov-report=html

lint:
	python3 -m flake8 src/ tests/ --max-line-length=120

run:
	python3 -m src.cli start

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov
