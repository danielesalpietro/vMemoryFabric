"""Issue #33 — unit test per benchmarks/perf_test_hardware.py.

Copre le parti pure/deterministiche (percentili, lettura cgroup CPU,
flag AVX da /proc/cpuinfo) con lo stesso pattern di mocking di
TestReadCgroupAvailableGb (test_scheduler.py — il gemello RAM di
_read_cgroup_cpu_count(), stesso principio v2->v1->fallback). Le
sezioni bench_*() sono coperte come smoke test end-to-end a dimensioni
sintetiche di default (piccole, veloci) — non un benchmark con
percentili affidabili, quello vive nello script stesso quando eseguito
su hardware reale (stesso principio di TestThroughputSmoke in
test_cpu_kernel.py: soglie larghe/strutturali, non numeri di
performance attesi).
"""
from __future__ import annotations

import io

import pytest
import torch

import benchmarks.perf_test_hardware as pth
from benchmarks.perf_test_hardware import (
    _bench_matmul_gflops,
    _cpu_flags,
    _percentiles,
    _read_cgroup_cpu_count,
    bench_cpu,
    bench_gpu,
    bench_pcie,
    bench_ram,
)


class TestPercentiles:
    def test_empty_list_returns_none_everywhere(self):
        assert _percentiles([]) == {"p50_ms": None, "p95_ms": None, "p99_ms": None}

    def test_percentiles_from_known_sorted_list(self):
        # 10 valori 0.001..0.010s -> indice p50=5 (0.006s), p95/p99=indice 9 (0.010s)
        latencies_s = [i / 1000 for i in range(1, 11)]
        pct = _percentiles(latencies_s)
        assert pct["p50_ms"] == pytest.approx(6.0)
        assert pct["p95_ms"] == pytest.approx(10.0)
        assert pct["p99_ms"] == pytest.approx(10.0)

    def test_percentiles_independent_of_input_order(self):
        latencies_s = [0.005, 0.001, 0.010, 0.003]
        assert _percentiles(latencies_s) == _percentiles(list(reversed(latencies_s)))


class TestReadCgroupCpuCount:
    """Gemello CPU di TestReadCgroupAvailableGb (test_scheduler.py) — stesso
    principio: v2 -> v1 -> fallback a os.cpu_count(), mai fidarsi di
    `nproc` grezzo quando un cgroup reale è leggibile
    (BOOTSTRAP_ANTI_ALZHEIMER.md §5.9, 'nproc' mente su RunPod)."""

    def test_reads_cgroup_v2_quota(self, monkeypatch):
        def _fake_open(path, *a, **kw):
            if str(path) == "/sys/fs/cgroup/cpu.max":
                return io.StringIO("272000 100000")
            raise FileNotFoundError(path)

        monkeypatch.setattr("builtins.open", _fake_open)

        assert _read_cgroup_cpu_count() == pytest.approx(2.72)

    def test_v2_max_sentinel_falls_through_to_v1(self, monkeypatch):
        def _fake_open(path, *a, **kw):
            path = str(path)
            if path == "/sys/fs/cgroup/cpu.max":
                return io.StringIO("max")
            if path == "/sys/fs/cgroup/cpu/cpu.cfs_quota_us":
                return io.StringIO("50000")
            if path == "/sys/fs/cgroup/cpu/cpu.cfs_period_us":
                return io.StringIO("100000")
            raise FileNotFoundError(path)

        monkeypatch.setattr("builtins.open", _fake_open)

        assert _read_cgroup_cpu_count() == pytest.approx(0.5)

    def test_v1_no_limit_sentinel_falls_through_to_os_cpu_count(self, monkeypatch):
        """cfs_quota_us == -1 e' il sentinel "nessun limite" cgroup v1 -
        deve cadere sul fallback os.cpu_count(), non un rapporto negativo."""
        def _fake_open(path, *a, **kw):
            path = str(path)
            if path == "/sys/fs/cgroup/cpu.max":
                raise FileNotFoundError(path)
            if path == "/sys/fs/cgroup/cpu/cpu.cfs_quota_us":
                return io.StringIO("-1")
            if path == "/sys/fs/cgroup/cpu/cpu.cfs_period_us":
                return io.StringIO("100000")
            raise FileNotFoundError(path)

        monkeypatch.setattr("builtins.open", _fake_open)
        monkeypatch.setattr(pth.os, "cpu_count", lambda: 16)

        assert _read_cgroup_cpu_count() == 16.0

    def test_falls_back_to_os_cpu_count_when_nothing_readable(self, monkeypatch):
        def _always_missing(path, *a, **kw):
            raise FileNotFoundError(path)

        monkeypatch.setattr("builtins.open", _always_missing)
        monkeypatch.setattr(pth.os, "cpu_count", lambda: 8)

        assert _read_cgroup_cpu_count() == 8.0

    def test_returns_none_when_os_cpu_count_also_unavailable(self, monkeypatch):
        def _always_missing(path, *a, **kw):
            raise FileNotFoundError(path)

        monkeypatch.setattr("builtins.open", _always_missing)
        monkeypatch.setattr(pth.os, "cpu_count", lambda: None)

        assert _read_cgroup_cpu_count() is None


class TestCpuFlags:
    def test_parses_flags_line_from_proc_cpuinfo(self, monkeypatch):
        cpuinfo = "processor\t: 0\nflags\t\t: fpu vme avx2 avx512f\nmodel name\t: x\n"

        def _fake_open(path, *a, **kw):
            if str(path) == "/proc/cpuinfo":
                return io.StringIO(cpuinfo)
            raise FileNotFoundError(path)

        monkeypatch.setattr("builtins.open", _fake_open)

        flags = _cpu_flags()
        assert "avx2" in flags
        assert "avx512f" in flags
        assert "avx512vnni" not in flags

    def test_returns_empty_list_when_unreadable(self, monkeypatch):
        def _always_missing(path, *a, **kw):
            raise FileNotFoundError(path)

        monkeypatch.setattr("builtins.open", _always_missing)

        assert _cpu_flags() == []


class TestBenchMatmulGflops:
    """Dimensioni sintetiche di default (hidden=512/intermediate=1536,
    OSX_BENCH_HIDDEN/INTERMEDIATE non impostate) - deliberatamente
    piccole per restare veloce ovunque, stesso principio di
    bench_cpu_kernel.py. Un run a dimensioni Mixtral reali va fatto
    manualmente sull'hardware target (vedi Usage nel modulo), non qui."""

    def test_cpu_structure_and_positive_gflops(self):
        result = _bench_matmul_gflops("cpu")

        assert result["hidden"] == pth._HIDDEN
        assert result["intermediate"] == pth._INTERMEDIATE
        for batch_key in ("batch_1", "batch_8", "batch_32"):
            assert result[batch_key]["gflops_at_p50"] > 0
            assert result[batch_key]["latency"]["p50_ms"] is not None


class TestBenchCpu:
    def test_structure(self):
        result = bench_cpu()

        assert result["status"] == "done"
        assert "cgroup_cores_available" in result
        assert result["os_cpu_count_raw"] == pth.os.cpu_count()
        assert set(result["avx_support"]) == {"avx2", "avx512f", "avx512_vnni"}
        assert "matmul_gflops" in result


class TestBenchRam:
    def test_structure_with_mocked_cgroup_reader(self, monkeypatch):
        monkeypatch.setattr(pth, "_read_cgroup_available_gb", lambda: 123.4)

        result = bench_ram()

        assert result["status"] == "done"
        assert result["cgroup_available_gb"] == 123.4
        assert result["copy_bandwidth_gbps"] > 0
        assert result["copy_test_size_mb"] > 0

    def test_none_available_gb_when_no_cgroup(self, monkeypatch):
        monkeypatch.setattr(pth, "_read_cgroup_available_gb", lambda: None)

        result = bench_ram()

        assert result["cgroup_available_gb"] is None


class TestBenchGpu:
    def test_skipped_without_cuda(self, monkeypatch):
        monkeypatch.setattr(pth.torch.cuda, "is_available", lambda: False)

        assert bench_gpu() == {
            "status": "skipped",
            "reason": "CUDA non disponibile su questo host",
        }

    @pytest.mark.gpu
    def test_done_with_real_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA non disponibile su questo host")

        result = bench_gpu()

        assert result["status"] == "done"
        assert result["vram_total_gb"] > 0
        assert result["vram_free_gb"] >= 0
        assert "matmul_gflops" in result

        # Rilascia i blocchi cached dell'allocatore PyTorch al driver -
        # trovato eseguendo la suite completa: senza questo,
        # test_tier.py::test_vram_free_decreases_after_load (NVML free
        # bytes PRIMA/DOPO un'allocazione di 64MB) può fallire se gira
        # DOPO questo test nello stesso processo, perché l'allocazione
        # successiva viene soddisfatta dalla cache di PyTorch invece che
        # da un cudaMalloc reale — NVML non vede alcuna variazione. Non
        # un bug nel codice di produzione, solo interazione tra ordine
        # dei test e stato dell'allocatore CUDA globale al processo.
        torch.cuda.empty_cache()


class TestBenchPcie:
    def test_skipped_without_cuda(self, monkeypatch):
        monkeypatch.setattr(pth.torch.cuda, "is_available", lambda: False)

        assert bench_pcie() == {
            "status": "skipped",
            "reason": "CUDA non disponibile su questo host",
        }

    @pytest.mark.gpu
    def test_done_with_real_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA non disponibile su questo host")

        result = bench_pcie()

        assert result["status"] == "done"
        assert set(result["by_size"]) == {"1mb", "16mb", "64mb", "256mb"}
        for entry in result["by_size"].values():
            assert entry["h2d_pinned_gbps"] > 0
            assert entry["d2h_pinned_gbps"] > 0

        # Stesso motivo di TestBenchGpu.test_done_with_real_gpu sopra -
        # bench_pcie() alloca/libera più tensori pinned pageable di
        # varie dimensioni, lasciando una cache dell'allocatore PyTorch
        # che altrimenti maschera un'allocazione NVML-visibile in un
        # test successivo (test_tier.py).
        torch.cuda.empty_cache()
