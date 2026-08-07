# OSX-PoC Makefile
# Tutti i target girano nel container via docker compose run.
# Prerequisito: docker compose build (fatto una volta sola).

COMPOSE  = docker compose
SERVICE  = osx-dev
RUN      = $(COMPOSE) run --rm $(SERVICE)

.PHONY: build smoke test bench lint typecheck shell clean

# ── Setup ──────────────────────────────────────────────────────────────────────

build:
	$(COMPOSE) build

# ── Smoke test (giorno 1) ─────────────────────────────────────────────────────

smoke:
	$(RUN) python scripts/smoke_test.py

# ── Test ──────────────────────────────────────────────────────────────────────

test:
	$(RUN) pytest tests/ -v --tb=short \
	    --cov=src --cov-report=term-missing \
	    --asyncio-mode=auto

test-eat:
	$(RUN) pytest tests/test_eat.py -v --tb=short

test-tier:
	$(RUN) pytest tests/test_tier.py -v --tb=short --asyncio-mode=auto

test-scheduler:
	$(RUN) pytest tests/test_scheduler.py -v --tb=short

# ── Benchmarks ────────────────────────────────────────────────────────────────

bench:
	$(RUN) python benchmarks/bench_eat.py
	$(RUN) python benchmarks/bench_tier.py

bench-eat:
	$(RUN) python benchmarks/bench_eat.py

bench-tier:
	$(RUN) python benchmarks/bench_tier.py

# ── Lint / Typecheck ──────────────────────────────────────────────────────────

lint:
	$(RUN) ruff check src/ tests/ scripts/

lint-fix:
	$(RUN) ruff check --fix src/ tests/ scripts/

typecheck:
	$(RUN) mypy src/ --ignore-missing-imports

# ── Dev shell ─────────────────────────────────────────────────────────────────

shell:
	$(COMPOSE) run --rm -it $(SERVICE) /bin/bash

# ── Metrics stack ─────────────────────────────────────────────────────────────

metrics-up:
	$(COMPOSE) --profile metrics up -d prometheus

metrics-down:
	$(COMPOSE) --profile metrics down

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"   -exec rm -rf {} + 2>/dev/null || true

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "OSX-PoC Makefile"
	@echo "────────────────────────────────────────────────"
	@echo "  make build        Build Docker image"
	@echo "  make smoke        Smoke test hardware + env"
	@echo "  make test         pytest tutti i moduli"
	@echo "  make test-eat     pytest M1 only"
	@echo "  make test-tier    pytest M2 only"
	@echo "  make test-scheduler pytest M3 only"
	@echo "  make bench        Benchmark M1 + M2"
	@echo "  make lint         Ruff linter"
	@echo "  make typecheck    mypy"
	@echo "  make shell        Shell interattiva nel container"
	@echo "  make metrics-up   Avvia Prometheus"
	@echo "  make clean        Rimuovi cache Python"
	@echo ""
