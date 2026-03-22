# =============================================================================
# SwingTrader — Makefile
# =============================================================================
# Provides convenient targets for development, testing, and deployment.
# All targets assume the project root is the working directory.
#
# Quick start:
#   make setup       # Create virtualenv and install all dependencies
#   make db-init     # Initialise PostgreSQL schema
#   make run         # Start bot in paper-trading mode
# =============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Project variables
# ---------------------------------------------------------------------------
PYTHON      ?= python3
VENV        := .venv
VENV_PYTHON := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip
PYTEST      := $(VENV)/bin/pytest
RUFF        := $(VENV)/bin/ruff
MYPY        := $(VENV)/bin/mypy
ALEMBIC     := $(VENV)/bin/alembic

IMAGE_NAME  := swingtrader
IMAGE_TAG   := latest

# Detect OS for platform-specific adjustments
OS := $(shell uname -s)

# ---------------------------------------------------------------------------
# Virtualenv & dependency management
# ---------------------------------------------------------------------------

.PHONY: setup
setup: ## Create virtualenv and install all runtime dependencies
	@echo "==> Creating virtualenv at $(VENV) ..."
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .
	@echo ""
	@echo "==> Setup complete."
	@echo "    Activate the virtualenv with:  source $(VENV)/bin/activate"

.PHONY: dev
dev: setup ## Install dev dependencies (testing, linting, type-checking)
	@echo "==> Installing dev dependencies ..."
	$(PIP) install -e ".[dev]"
	@echo "==> Dev environment ready."

.PHONY: install
install: ## Install only runtime dependencies (no virtualenv creation)
	$(PIP) install -e .

# ---------------------------------------------------------------------------
# Running the bot
# ---------------------------------------------------------------------------

.PHONY: run
run: ## Start the bot in PAPER trading mode
	@echo "==> Starting SwingTrader (paper mode) ..."
	TRADING_MODE=paper $(VENV_PYTHON) -m src.main

.PHONY: run-live
run-live: ## Start the bot in LIVE trading mode (⚠️  real money)
	@echo ""
	@echo "⚠️  WARNING: Live trading uses REAL MONEY."
	@echo "   Press Ctrl+C within 5 seconds to cancel ..."
	@echo ""
	@sleep 5
	TRADING_MODE=live $(VENV_PYTHON) -m src.main

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

.PHONY: test
test: ## Run the full test suite with pytest
	@echo "==> Running tests ..."
	$(PYTEST) tests/ -v --tb=short

.PHONY: test-unit
test-unit: ## Run unit tests only
	$(PYTEST) tests/unit/ -v --tb=short

.PHONY: test-integration
test-integration: ## Run integration tests only
	$(PYTEST) tests/integration/ -v --tb=short

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	$(PYTEST) tests/ -v --tb=short \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-fail-under=70
	@echo "==> Coverage report: htmlcov/index.html"

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

.PHONY: lint
lint: ## Run ruff linter
	@echo "==> Running ruff ..."
	$(RUFF) check src/ tests/ scripts/
	@echo "==> Lint passed."

.PHONY: lint-fix
lint-fix: ## Run ruff linter with auto-fix
	$(RUFF) check --fix src/ tests/ scripts/

.PHONY: format
format: ## Format code with ruff formatter
	$(RUFF) format src/ tests/ scripts/

.PHONY: typecheck
typecheck: ## Run mypy type checker
	@echo "==> Running mypy ..."
	$(MYPY) src/
	@echo "==> Type check passed."

.PHONY: check
check: lint typecheck ## Run all code quality checks (lint + typecheck)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

.PHONY: db-init
db-init: ## Initialise the database schema and seed config
	@echo "==> Initialising database ..."
	$(VENV_PYTHON) scripts/init_db.py

.PHONY: db-migrate
db-migrate: ## Apply all pending Alembic migrations
	@echo "==> Applying Alembic migrations ..."
	$(ALEMBIC) upgrade head

.PHONY: db-migrate-down
db-migrate-down: ## Roll back the most recent Alembic migration
	@echo "==> Rolling back last migration ..."
	$(ALEMBIC) downgrade -1

.PHONY: db-revision
db-revision: ## Generate a new Alembic migration (auto-detect changes)
	@read -p "Migration message: " msg; \
	$(ALEMBIC) revision --autogenerate -m "$$msg"

.PHONY: db-history
db-history: ## Show Alembic migration history
	$(ALEMBIC) history --verbose

.PHONY: db-current
db-current: ## Show current Alembic revision
	$(ALEMBIC) current

.PHONY: seed-universe
seed-universe: ## Seed the tradable stock universe (hardcoded top-100 S&P 500)
	@echo "==> Seeding stock universe ..."
	$(VENV_PYTHON) scripts/seed_universe.py

.PHONY: seed-universe-fetch
seed-universe-fetch: ## Seed universe and fetch fresh data from Alpaca API
	$(VENV_PYTHON) scripts/seed_universe.py --fetch

# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

.PHONY: backtest
backtest: ## Run backtesting script (requires --start and --end args)
	@echo "==> Running backtest ..."
	@echo "    Usage: make backtest ARGS='--start 2023-01-01 --end 2023-12-31'"
	$(VENV_PYTHON) scripts/run_backtest.py $(ARGS)

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

.PHONY: docker-build
docker-build: ## Build the Docker image (runtime stage)
	@echo "==> Building Docker image $(IMAGE_NAME):$(IMAGE_TAG) ..."
	docker build \
		--target runtime \
		--tag $(IMAGE_NAME):$(IMAGE_TAG) \
		--file Dockerfile \
		.
	@echo "==> Image built: $(IMAGE_NAME):$(IMAGE_TAG)"

.PHONY: docker-build-no-cache
docker-build-no-cache: ## Build Docker image without layer cache
	docker build \
		--no-cache \
		--target runtime \
		--tag $(IMAGE_NAME):$(IMAGE_TAG) \
		--file Dockerfile \
		.

.PHONY: docker-up
docker-up: ## Start all services with Docker Compose
	@echo "==> Starting Docker Compose stack ..."
	docker compose up -d
	@echo ""
	@echo "==> Services started."
	@echo "    API:      http://localhost:8000"
	@echo "    API docs: http://localhost:8000/docs (paper mode only)"
	@echo "    Metrics:  http://localhost:9090/metrics"
	@echo "    Logs:     docker compose logs -f app"

.PHONY: docker-up-build
docker-up-build: ## Build and start all services with Docker Compose
	docker compose up -d --build

.PHONY: docker-down
docker-down: ## Stop and remove all Docker Compose containers
	@echo "==> Stopping Docker Compose stack ..."
	docker compose down

.PHONY: docker-down-volumes
docker-down-volumes: ## Stop containers AND remove all named volumes (data loss!)
	@echo "⚠️  WARNING: This will delete all persistent data (postgres, redis)."
	@read -p "Are you sure? [y/N] " confirm; \
	if [ "$$confirm" = "y" ]; then docker compose down -v; fi

.PHONY: docker-logs
docker-logs: ## Tail logs from the app container
	docker compose logs -f app

.PHONY: docker-shell
docker-shell: ## Open a shell inside the running app container
	docker compose exec app /bin/bash

.PHONY: docker-ps
docker-ps: ## Show status of all Docker Compose services
	docker compose ps

.PHONY: docker-restart
docker-restart: ## Restart the app container
	docker compose restart app

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

.PHONY: clean
clean: ## Remove caches, compiled files, test artefacts
	@echo "==> Cleaning build artefacts ..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "==> Clean done."

.PHONY: env-check
env-check: ## Verify required environment variables are set
	@echo "==> Checking required environment variables ..."
	@missing=0; \
	for var in ALPACA_API_KEY ALPACA_API_SECRET DATABASE_URL REDIS_URL API_KEY; do \
		if [ -z "$${!var}" ]; then \
			echo "  [MISSING] $$var"; \
			missing=1; \
		else \
			echo "  [OK]      $$var"; \
		fi; \
	done; \
	if [ $$missing -eq 1 ]; then \
		echo ""; \
		echo "  Some variables are missing. Copy .env.example to .env and fill them in."; \
		exit 1; \
	fi
	@echo "==> All required variables are set."

.PHONY: help
help: ## Show this help message
	@echo ""
	@echo "SwingTrader — Make Targets"
	@echo "=========================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' | \
		sort
	@echo ""
