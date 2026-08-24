"""Issue #33 — unit test per benchmarks/perf_tuning_report.py.

Tutte le funzioni sotto test sono pure (dict JSON -> dict di
raccomandazioni), nessun hardware reale richiesto — a differenza di
perf_test_hardware.py (vedi test_perf_test_hardware.py), qui il JSON di
input è costruito a mano per ogni scenario, stesso principio delle
fixture sintetiche già usate altrove nel progetto (es.
TestCheckCpuRamBudget in test_scheduler.py).
"""
from __future__ import annotations

import pytest

from benchmarks.perf_tuning_report import (
    _PER_EXPERT_CPU_GB_AWQ,
    _PER_EXPERT_CPU_GB_PATH1,
    _RAM_MARGIN_GB_DEFAULT,
    _batching_signal,
    _cpu_offload_verdict,
    _pcie_flag,
    _ram_budget,
    _thread_count_recommendation,
    build_report,
)


def _matmul_section(batch1, batch8=None, batch32=None):
    """Stessa forma di _bench_matmul_gflops() in perf_test_hardware.py —
    solo gflops_at_p50 è rilevante per la logica sotto test, la latenza
    è un placeholder."""
    return {
        "hidden": 512,
        "intermediate": 1536,
        "n_repeats": 30,
        "batch_1": {"latency": {"p50_ms": 1.0}, "gflops_at_p50": batch1},
        "batch_8": {"latency": {"p50_ms": 1.0}, "gflops_at_p50": batch8 or batch1},
        "batch_32": {"latency": {"p50_ms": 1.0}, "gflops_at_p50": batch32 or batch1},
    }


class TestCpuOffloadVerdict:
    def test_predicted_slowdown_matches_measured_ratio(self):
        hw = {
            "cpu": {"status": "done", "matmul_gflops": _matmul_section(10.0),
                    "avx_support": {"avx512f": True}},
            "gpu": {"status": "done", "matmul_gflops": _matmul_section(250.0)},
        }

        result = _cpu_offload_verdict(hw)

        assert result["predicted_slowdown_factor"] == pytest.approx(25.0)
        assert "coerente" in result["consistency_with_known_data"]
        assert result["avx512f_present"] is True
        assert result["avx512_note"] is None

    def test_flags_out_of_range_when_far_from_known_data(self):
        # rapporto 1.25x — molto sotto il range 24-26x già osservato
        # (esattamente il caso reale misurato sullo Z8 a dimensioni
        # sintetiche piccole, LOGBOOK_ISSUE33.MD "continued 19").
        hw = {
            "cpu": {"status": "done", "matmul_gflops": _matmul_section(200.0),
                    "avx_support": {"avx512f": True}},
            "gpu": {"status": "done", "matmul_gflops": _matmul_section(250.0)},
        }

        result = _cpu_offload_verdict(hw)

        assert "FUORI" in result["consistency_with_known_data"]

    def test_avx512_absent_adds_explicit_note(self):
        hw = {
            "cpu": {"status": "done", "matmul_gflops": _matmul_section(10.0),
                    "avx_support": {"avx512f": False}},
            "gpu": {"status": "done", "matmul_gflops": _matmul_section(250.0)},
        }

        result = _cpu_offload_verdict(hw)

        assert result["avx512f_present"] is False
        assert result["avx512_note"] is not None

    def test_na_without_gpu_section(self):
        hw = {
            "cpu": {"status": "done", "matmul_gflops": _matmul_section(10.0)},
            "gpu": {"status": "skipped", "reason": "no cuda"},
        }

        assert _cpu_offload_verdict(hw)["verdict"] == "n/a"

    def test_na_without_cpu_section(self):
        hw = {
            "cpu": {"status": "skipped"},
            "gpu": {"status": "done", "matmul_gflops": _matmul_section(250.0)},
        }

        assert _cpu_offload_verdict(hw)["verdict"] == "n/a"

    def test_na_when_gflops_missing(self):
        hw = {
            "cpu": {"status": "done", "matmul_gflops": _matmul_section(None),
                    "avx_support": {"avx512f": True}},
            "gpu": {"status": "done", "matmul_gflops": _matmul_section(250.0)},
        }

        assert _cpu_offload_verdict(hw)["verdict"] == "n/a"


class TestBatchingSignal:
    def test_high_ratio_signals_batching_opportunity(self):
        hw = {"cpu": {"status": "done", "matmul_gflops": _matmul_section(10.0, batch32=100.0)}}

        result = _batching_signal(hw)

        assert result["gflops_ratio_batch32_over_batch1"] == pytest.approx(10.0)
        assert "potrebbe recuperare" in result["signal"]

    def test_low_ratio_signals_no_meaningful_gain(self):
        hw = {"cpu": {"status": "done", "matmul_gflops": _matmul_section(10.0, batch32=11.0)}}

        result = _batching_signal(hw)

        assert "nessun guadagno" in result["signal"]

    def test_na_without_cpu_section(self):
        assert _batching_signal({"cpu": {"status": "skipped"}})["signal"] == "n/a"


class TestThreadCountRecommendation:
    def test_recommends_all_cgroup_cores_for_pool_build_only(self):
        hw = {"cpu": {"status": "done", "cgroup_cores_available": 27.2}}

        result = _thread_count_recommendation(hw)

        assert result["omp_mkl_num_threads_for_pool_build"] == 27
        assert "NON aumentare" in result["note_per_token_forward"]

    def test_none_when_cgroup_cores_undetermined(self):
        hw = {"cpu": {"status": "done", "cgroup_cores_available": None}}

        result = _thread_count_recommendation(hw)

        assert result["omp_mkl_num_threads_for_pool_build"] is None

    def test_na_without_cpu_section(self):
        assert _thread_count_recommendation({"cpu": {"status": "skipped"}})["recommendation"] == "n/a"


class TestRamBudget:
    """Stesse costanti di GCSGWorker._check_cpu_ram_budget()
    (scheduler/gcsg.py), duplicate deliberatamente in
    perf_tuning_report.py — vedi il commento lì sul perché."""

    def test_computes_max_experts_both_paths(self):
        hw = {"ram": {"status": "done", "cgroup_available_gb": 116.0}}

        result = _ram_budget(hw)

        expected_budget = 116.0 - _RAM_MARGIN_GB_DEFAULT
        assert result["usable_budget_gb"] == pytest.approx(expected_budget)
        assert result["max_cpu_shadow_experts_path1_int4"] == int(
            expected_budget // _PER_EXPERT_CPU_GB_PATH1,
        )
        assert result["max_cpu_shadow_experts_path_awq_fp32"] == int(
            expected_budget // _PER_EXPERT_CPU_GB_AWQ,
        )

    def test_zero_experts_when_budget_negative(self):
        # 10GB disponibili < 24GB margine di default -> budget negativo,
        # nessun expert entra (stesso principio "degrada, mai un numero
        # negativo" di GCSGWorker._check_cpu_ram_budget()).
        hw = {"ram": {"status": "done", "cgroup_available_gb": 10.0}}

        result = _ram_budget(hw)

        assert result["max_cpu_shadow_experts_path1_int4"] == 0
        assert result["max_cpu_shadow_experts_path_awq_fp32"] == 0

    def test_na_when_ram_undeterminable(self):
        hw = {"ram": {"status": "done", "cgroup_available_gb": None}}

        assert _ram_budget(hw)["recommendation"] == "n/a"

    def test_na_without_ram_section(self):
        assert _ram_budget({"ram": {"status": "skipped"}})["recommendation"] == "n/a"


class TestPcieFlag:
    def test_ok_when_pinned_bandwidth_healthy(self):
        hw = {"pcie": {"status": "done", "by_size": {
            "1mb": {"h2d_pinned_gbps": 11.0, "d2h_pinned_gbps": 11.5,
                    "h2d_pageable_gbps": 2.0, "d2h_pageable_gbps": 2.0},
            "256mb": {"h2d_pinned_gbps": 12.0, "d2h_pinned_gbps": 11.2,
                      "h2d_pageable_gbps": 4.0, "d2h_pageable_gbps": 3.0},
        }}}

        result = _pcie_flag(hw)

        assert result["flag"] == "ok"
        assert result["pinned_min_gbps"] == pytest.approx(11.2)

    def test_flags_suspect_when_pinned_bandwidth_low_on_largest_size(self):
        hw = {"pcie": {"status": "done", "by_size": {
            "1mb": {"h2d_pinned_gbps": 11.0, "d2h_pinned_gbps": 11.5,
                    "h2d_pageable_gbps": 2.0, "d2h_pageable_gbps": 2.0},
            "256mb": {"h2d_pinned_gbps": 2.0, "d2h_pinned_gbps": 1.5,
                      "h2d_pageable_gbps": 1.0, "d2h_pageable_gbps": 1.0},
        }}}

        result = _pcie_flag(hw)

        assert result["flag"] == "SOSPETTO"
        assert "WSL2/GPU-PV" in result["reason"]

    def test_na_when_skipped(self):
        hw = {"pcie": {"status": "skipped", "reason": "CUDA non disponibile"}}

        result = _pcie_flag(hw)

        assert result["flag"] == "n/a"
        assert result["reason"] == "CUDA non disponibile"


class TestBuildReport:
    def test_combines_all_sections(self):
        hw = {
            "host": "test-host",
            "cpu": {"status": "done", "matmul_gflops": _matmul_section(10.0),
                    "cgroup_cores_available": 8.0, "avx_support": {"avx512f": True}},
            "gpu": {"status": "done", "matmul_gflops": _matmul_section(250.0)},
            "ram": {"status": "done", "cgroup_available_gb": 116.0},
            "pcie": {"status": "skipped", "reason": "CUDA non disponibile"},
        }

        report = build_report(hw)

        assert report["status"] == "done"
        assert report["source_host"] == "test-host"
        assert set(report) == {
            "status", "issue", "source_host", "cpu_offload_verdict",
            "thread_count", "ram_budget", "pcie", "batching_signal",
        }
        assert report["cpu_offload_verdict"]["predicted_slowdown_factor"] == pytest.approx(25.0)
