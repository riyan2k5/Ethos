.PHONY: help install test lint format validate train clean docker-build docker-run

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	pip install -r requirements.txt

test: ## Run all tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term

lint: ## Run linters
	flake8 . --max-line-length=127
	pylint ml/ preprocessing/ scripts/ || true

format: ## Format code with Black
	black .

format-check: ## Check code formatting
	black --check .

validate: ## Validate data
	python scripts/validate_data.py --data-path data/spotify_data_reduced.csv --target-column genre

train: ## Train the genre classification model
	cd ml && python train_genre_model.py

train-all: ## Train all ML models
	python scripts/train_all_models.py

clean: ## Clean cache and temporary files
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov/ dist/ build/

docker-build-training: ## Build training Docker image
	docker build --target training -t ethos:training .

docker-build-production: ## Build production Docker image
	docker build --target production -t ethos:production .

docker-run-training: ## Run training container
	docker run -v $(PWD)/data:/app/data ethos:training

ci: format-check lint test ## Run all CI checks locally
