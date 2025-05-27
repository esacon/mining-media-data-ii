.PHONY: help format lint test clean install check all

# Default target
help:
	@echo "Available commands:"
	@echo "  install     Install dependencies"
	@echo "  format      Format code with black and isort"
	@echo "  lint        Run flake8 linter"
	@echo "  check       Run format check without making changes"
	@echo "  test        Run pytest tests"
	@echo "  clean       Clean up cache files"
	@echo "  all         Run format, lint, and test"

# Install dependencies
install:
	pip install -r requirements.txt

# Format code with black and isort
format:
	@echo "Running isort..."
	isort src/ --diff --color
	isort src/
	@echo "Running black..."
	black src/ --diff --color
	black src/

# Check formatting without making changes
check:
	@echo "Checking isort..."
	isort src/ --check-only --diff --color
	@echo "Checking black..."
	black src/ --check --diff --color

# Run linter
lint:
	@echo "Running flake8..."
	flake8 src/

# Run tests
test:
	@echo "Running pytest..."
	pytest

# Clean up cache files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

# Run all checks
all: format lint test
	@echo "All checks completed!"

# Pre-commit hook simulation
pre-commit: check lint
	@echo "Pre-commit checks passed!" 