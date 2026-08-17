# OSX-PoC Makefile
# Tutti i target girano nel container via docker compose run.
# Prerequisito: docker compose build (fatto una volta sola).

COMPOSE  = docker compose
SERVICE  = osx-dev
RUN      = $(COMPOSE) run --rm $(SERVICE)

.PHONY: build smoke test bench lint typecheck shell clean gpu-repro-issue10

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

test-cpu-kernel:
	$(RUN) pytest tests/test_cpu_kernel.py -v --tb=short

# ── Benchmarks ────────────────────────────────────────────────────────────────

bench:
	$(RUN) python benchmarks/bench_eat.py
	$(RUN) python benchmarks/bench_tier.py

bench-eat:
	$(RUN) python benchmarks/bench_eat.py

bench-tier:
	$(RUN) python benchmarks/bench_tier.py

bench-cpu-kernel:
	$(RUN) python benchmarks/bench_cpu_kernel.py

bench-route-forward:
	$(RUN) python benchmarks/bench_route_forward.py

bench-hybrid:
	$(RUN) python benchmarks/bench_hybrid.py

# ── Performance framework (issue #33, riutilizzabile oltre l'issue) ────────────
# perf-test-hardware caratterizza CPU/RAM/GPU/PCIe di questo host/pod;
# perf-tuning-report legge quel JSON e produce raccomandazioni (thread count,
# budget RAM CPU-pool, verdetto cpu-offload) — vedi i docstring dei due script
# per come sono ancorati ai dati già misurati in LOGBOOK_ISSUE33.MD.

perf-test-hardware:
	$(RUN) python benchmarks/perf_test_hardware.py

perf-tuning-report:
	$(RUN) sh -c "python benchmarks/perf_test_hardware.py | python benchmarks/perf_tuning_report.py"

# ── Diagnostic repros (manual, not wired into CI) ───────────────────────────────
# Requires the real checkpoint on disk (/data/nvme/models/mixtral-instruct-awq)
# and a real GPU — not CI-safe (some variants are designed to crash, that's the
# signal). See LOGBOOK.md 2026-08-10 for the full trail. Each script takes
# --variant; run without one to see the choices.

gpu-repro-issue10:
	@echo "issue #10 / path-3 offload crash — diagnostic repros, one variant per run:"
	@echo "  $(RUN) python scripts/isolate_awq_shadow_call_crash.py"
	@echo "  $(RUN) python scripts/isolate_awq_offload_variables.py --variant {non-offloaded,pin-memory-forced}"
	@echo "  $(RUN) python scripts/isolate_marlin_offload_variables.py --variant {non-offloaded,offloaded-unpinned,offloaded-pinned}"
	@echo "  $(RUN) python scripts/verify_awq_manual_shadow_expert.py"
	@echo "Run one at a time — CUDA errors are sticky for the rest of a process."

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
	@echo "  make test-cpu-kernel pytest issue #33 Fase 1 (CPU kernel, no GPU needed)"
	@echo "  make bench        Benchmark M1 + M2"
	@echo "  make bench-cpu-kernel  Benchmark issue #33 Fase 1 (CPU forward throughput)"
	@echo "  make perf-test-hardware  Caratterizza CPU/RAM/GPU/PCIe di questo host/pod (issue #33)"
	@echo "  make perf-tuning-report  Raccomandazioni (thread/RAM/cpu-offload) da perf-test-hardware"
	@echo "  make gpu-repro-issue10  Diagnostic repros, issue #10 offload crash (manual, one at a time)"
	@echo "  make lint         Ruff linter"
	@echo "  make typecheck    mypy"
	@echo "  make shell        Shell interattiva nel container"
	@echo "  make metrics-up   Avvia Prometheus"
	@echo "  make clean        Rimuovi cache Python"
	@echo ""
