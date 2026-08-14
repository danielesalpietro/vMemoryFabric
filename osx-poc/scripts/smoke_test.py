#!/usr/bin/env python3
"""OSX-PoC — Smoke Test

Verifica che l'ambiente di sviluppo sia funzionale prima di iniziare Sprint 1.
Ogni check è indipendente: un fallimento non blocca gli altri.

Usage:
    python scripts/smoke_test.py
    make smoke          (via Makefile)

Output: PASS / FAIL per ogni check + riepilogo finale.
Exit code: 0 se tutti i check passano, 1 se almeno uno fallisce.
"""
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

# ── helpers ────────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def _pass(msg: str) -> None:
    print(f"  {GREEN}✓ PASS{RESET}  {msg}")

def _fail(msg: str, detail: str = "") -> None:
    print(f"  {RED}✗ FAIL{RESET}  {msg}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"         {line}")

def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠ WARN{RESET}  {msg}")

def _section(title: str) -> None:
    print(f"\n── {title} {'─' * (50 - len(title))}")

# ── checks ────────────────────────────────────────────────────────────────────

def check_python_version() -> bool:
    import sys
    major, minor = sys.version_info.major, sys.version_info.minor
    ok = (major == 3 and minor >= 12)
    if ok:
        _pass(f"Python {major}.{minor}")
    else:
        _fail(f"Python {major}.{minor} — richiesto 3.12+")
    return ok


def check_torch() -> bool:
    try:
        import torch
        _pass(f"PyTorch {torch.__version__}")
        return True
    except ImportError as e:
        _fail("PyTorch non importabile", str(e))
        return False


def check_cuda() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            _fail("CUDA non disponibile (torch.cuda.is_available() = False)")
            _warn("Verifica: NVIDIA Container Toolkit installato? --gpus flag in docker run?")
            return False
        n = torch.cuda.device_count()
        for i in range(n):
            name  = torch.cuda.get_device_name(i)
            vram  = torch.cuda.get_device_properties(i).total_memory // (1024**3)
            _pass(f"GPU {i}: {name} — {vram} GB VRAM")
            if "3090" not in name and i == 0:
                _warn(f"GPU 0 attesa: RTX 3090 — trovata: {name}")
        return True
    except Exception:
        _fail("Errore check CUDA", traceback.format_exc())
        return False


def check_vram_24gb() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            _fail("VRAM check saltato — CUDA non disponibile")
            return False
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)
        if vram_gb >= 22:   # soglia: 22 GB (margine per overhead driver)
            _pass(f"VRAM device 0: {vram_gb:.1f} GB (atteso ~24 GB)")
            return True
        else:
            _fail(f"VRAM device 0: {vram_gb:.1f} GB — atteso ≥ 22 GB")
            return False
    except Exception:
        _fail("Errore check VRAM", traceback.format_exc())
        return False


def check_cuda_tensor_roundtrip() -> bool:
    """Alloca un tensore su GPU e verificane il valore — test DMA funzionante."""
    try:
        import torch
        if not torch.cuda.is_available():
            _fail("Tensor roundtrip saltato — CUDA non disponibile")
            return False
        t0 = time.perf_counter()
        x = torch.ones(1024, 1024, device="cuda:0", dtype=torch.float32)
        y = x.cpu().numpy()
        latency_ms = (time.perf_counter() - t0) * 1000
        assert y.mean() == 1.0, "Valore tensore inatteso"
        _pass(f"CUDA tensor roundtrip OK — latenza: {latency_ms:.1f} ms")
        if latency_ms > 100:
            _warn("Latenza > 100 ms — possibile overhead senza pinned memory (atteso su Docker/Windows)")
        return True
    except Exception:
        _fail("Errore tensor roundtrip", traceback.format_exc())
        return False


def check_nvme_volume() -> bool:
    """Verifica che il volume NVMe sia montato e scrivibile."""
    nvme_path = Path("/data/nvme")
    try:
        if not nvme_path.exists():
            _fail(f"Volume NVMe non trovato: {nvme_path}")
            _warn("Verifica volume Docker: 'nvme-data:/data/nvme' in docker-compose.yml")
            return False
        test_file = nvme_path / ".smoke_test"
        test_file.write_text("osx-poc smoke test")
        test_file.unlink()
        _pass(f"Volume NVMe OK: {nvme_path} (scrivibile)")
        return True
    except Exception as e:
        _fail(f"Errore accesso volume NVMe: {nvme_path}", str(e))
        return False


def check_vllm() -> bool:
    """vLLM è deliberatamente escluso dall'immagine base (vedi requirements-vllm.txt):
    i hook GCSG che se ne servono sono ancora stub NotImplementedError (Sprint 3).
    Non installato = atteso, non è un fallimento dello smoke test."""
    try:
        import vllm
        _pass(f"vLLM {vllm.__version__} importabile (requirements-vllm.txt installato)")
    except ImportError:
        _warn("vLLM non installato — atteso: escluso dall'immagine base fino a Sprint 3 "
              "(installare manualmente con requirements-vllm.txt quando servirà)")
    return True


def check_transformers() -> bool:
    try:
        import transformers
        _pass(f"Transformers {transformers.__version__}")
        return True
    except ImportError as e:
        _fail("Transformers non importabile", str(e))
        return False


def check_onnxruntime() -> bool:
    try:
        import onnxruntime as ort
        _pass(f"ONNXRuntime {ort.__version__} — providers: {ort.get_available_providers()}")
        return True
    except ImportError as e:
        _fail("ONNXRuntime non importabile", str(e))
        return False


def check_prometheus_client() -> bool:
    try:
        from importlib.metadata import version

        import prometheus_client  # noqa: F401 — import verifica disponibilità pacchetto
        _pass(f"prometheus_client {version('prometheus_client')}")
        return True
    except ImportError as e:
        _fail("prometheus_client non importabile", str(e))
        return False


def check_aiofiles() -> bool:
    try:
        from importlib.metadata import version

        import aiofiles  # noqa: F401 — import verifica disponibilità pacchetto
        _pass(f"aiofiles {version('aiofiles')} (proxy io_uring)")
        return True
    except ImportError as e:
        _fail("aiofiles non importabile", str(e))
        return False


def check_osx_src_importable() -> bool:
    """Verifica che il package src/ sia importabile (PYTHONPATH=/workspace/osx-poc/src)."""
    try:
        from eat.eat import (
            ExpertAccessTable,  # noqa: F401 — import verifica disponibilità pacchetto
        )
        from eat.types import EATEntry, Tier  # noqa: F401
        from scheduler.ptpep import PTPEPClassifier  # noqa: F401
        from tier.manager import TierManager  # noqa: F401
        _pass("src/ packages importabili (eat, tier, scheduler)")
        return True
    except ImportError as e:
        _fail("src/ packages non importabili", str(e))
        _warn("Verifica: PYTHONPATH=/workspace/osx-poc/src impostato nel container")
        return False


def check_pinned_memory_absent() -> bool:
    """Verifica che pinned memory NON sia usata — constraint Docker/Windows."""
    try:
        import torch
        if not torch.cuda.is_available():
            _warn("Pinned memory check saltato — CUDA non disponibile")
            return True
        # Tentativo di alloc pinned memory: dovrebbe funzionare su Linux nativo,
        # ma può essere lento su Docker/Windows per via del driver virtualizzato.
        t0 = time.perf_counter()
        t = torch.zeros(256 * 1024 * 1024 // 4, dtype=torch.float32).pin_memory()
        latency_ms = (time.perf_counter() - t0) * 1000
        del t
        _warn(f"Pinned memory alloc: {latency_ms:.0f} ms — "
              f"{'LENTO (overhead Docker atteso)' if latency_ms > 500 else 'OK'}")
        _warn("OSX-PoC NON usa pinned memory per design (vincolo dev). "
              "cudaMemcpy standard viene usato al suo posto.")
        return True
    except Exception as e:
        _warn(f"Pinned memory non disponibile: {e} — comportamento atteso su Docker/Windows")
        return True   # non è un errore per il nostro setup


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("  OSX-PoC — Smoke Test")
    print("  Setup: Docker/Windows · RTX 3090 · Single GPU")
    print("=" * 60)

    checks: list[tuple[str, Callable[[], bool]]] = [
        ("Python version",          check_python_version),
        ("PyTorch",                 check_torch),
        ("CUDA availability",       check_cuda),
        ("VRAM ≥ 22 GB (3090)",     check_vram_24gb),
        ("CUDA tensor roundtrip",   check_cuda_tensor_roundtrip),
        ("NVMe volume",             check_nvme_volume),
        ("vLLM",                    check_vllm),
        ("Transformers",            check_transformers),
        ("ONNXRuntime",             check_onnxruntime),
        ("aiofiles (proxy io_uring)", check_aiofiles),
        ("prometheus_client",       check_prometheus_client),
        ("src/ importable",         check_osx_src_importable),
        ("Pinned memory (info)",    check_pinned_memory_absent),
    ]

    results: list[tuple[str, bool]] = []
    for name, fn in checks:
        _section(name)
        try:
            ok = fn()
        except Exception:
            _fail(name, traceback.format_exc())
            ok = False
        results.append((name, ok))

    # ── riepilogo ──────────────────────────────────────────────────────────────
    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed

    print("\n" + "=" * 60)
    print(f"  Riepilogo: {GREEN}{passed} PASS{RESET}  {RED}{failed} FAIL{RESET}")
    if failed > 0:
        print("\n  Check falliti:")
        for name, ok in results:
            if not ok:
                print(f"    {RED}✗{RESET} {name}")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
