#!/usr/bin/env python3
"""Issue #33 Fase 6a - misura diretta della banda PCIe CPU<->GPU sotto
WSL2, pinned vs pageable, indipendente da vLLM/MoE.

Nato da un dubbio sollevato durante la run cpu-offload: il traffico
osservato via `nvidia-smi dmon -s t` (~2.3 GB/s TX, ~800 MB/s RX)
potrebbe riflettere un tetto imposto da WSL2/GPU-PV (paravirtualizzazione,
niente GPUDirect RDMA), oppure semplicemente essere quanto il workload
attuale (vLLM cpu_offload_gb + route_forward()) genera senza saturare
nulla. I due scenari non si distinguono guardando solo il traffico di un
workload che non sta esplicitamente cercando di saturare il bus — serve
un microbenchmark dedicato che ci provi a saturarlo davvero.

Misura, per varie dimensioni di tensore, GB/s ottenuti con:
  - memoria pageable (torch.randn(...).to('cuda'), il default)
  - memoria pinned (torch.randn(...).pin_memory().to('cuda', non_blocking=True))
in entrambe le direzioni (H2D e D2H), con warmup + torch.cuda.synchronize()
per misure pulite.

Usage:
    python scripts/bench_pcie_bandwidth_wsl2.py
"""
from __future__ import annotations

import time

import torch


def _bench_h2d(size_bytes: int, pinned: bool, n_iters: int = 20) -> float:
    n_floats = size_bytes // 4
    cpu_tensor = torch.randn(n_floats, dtype=torch.float32)
    if pinned:
        cpu_tensor = cpu_tensor.pin_memory()

    # warmup
    _ = cpu_tensor.to("cuda", non_blocking=pinned)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = cpu_tensor.to("cuda", non_blocking=pinned)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (size_bytes * n_iters) / elapsed / 1e9   # GB/s


def _bench_d2h(size_bytes: int, pinned: bool, n_iters: int = 20) -> float:
    n_floats = size_bytes // 4
    gpu_tensor = torch.randn(n_floats, dtype=torch.float32, device="cuda")
    dst = torch.empty(n_floats, dtype=torch.float32)
    if pinned:
        dst = dst.pin_memory()

    _ = gpu_tensor.to("cpu" if not pinned else dst.device)  # warmup (semplice)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        dst.copy_(gpu_tensor, non_blocking=pinned)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (size_bytes * n_iters) / elapsed / 1e9


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA non disponibile — skip.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Gen3 x16 teorico: ~15.75 GB/s per direzione\n")

    sizes_mb = [1, 16, 64, 256, 1024]
    print(f"{'size':>8} | {'H2D pageable':>14} | {'H2D pinned':>12} | "
          f"{'D2H pageable':>14} | {'D2H pinned':>12}   (GB/s)")
    print("-" * 80)
    for mb in sizes_mb:
        size_bytes = mb * 1024 * 1024
        h2d_pageable = _bench_h2d(size_bytes, pinned=False)
        h2d_pinned = _bench_h2d(size_bytes, pinned=True)
        d2h_pageable = _bench_d2h(size_bytes, pinned=False)
        d2h_pinned = _bench_d2h(size_bytes, pinned=True)
        print(f"{mb:>6}MB | {h2d_pageable:>12.2f}   | {h2d_pinned:>10.2f}   | "
              f"{d2h_pageable:>12.2f}   | {d2h_pinned:>10.2f}")

    print("\nSe pinned >> pageable e si avvicina a ~12-15 GB/s: DMA reale "
          "funziona sotto WSL2, nessun tetto strutturale.")
    print("Se pinned non supera ~2-3 GB/s anche sui tensori grandi: "
          "supporto al sospetto di un collo di bottiglia WSL2/GPU-PV.")


if __name__ == "__main__":
    main()
