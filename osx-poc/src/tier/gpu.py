"""M2 — GPU Transfer (DDR4 → VRAM RTX 3090).

Adattamenti rispetto a spec OSX v1.0:
    - Pinned memory (cudaMallocHost) NON disponibile in Docker su Windows.
      Usiamo cudaMemcpy standard con pageable host memory.
      Il delta rispetto a pinned sarà documentato come baseline deviation.
    - Single GPU: solo RTX 3090 (device 0). Dual-GPU deferred.
    - Nessuna CUDA stream pipeline per ora — sarà aggiunta in Sprint 2
      per misurare l'overlap compute/transfer.

Target latenza DDR4 → VRAM (256 MB shard):
    Teorico pinned   : ~32 ms  (PCIe Gen3 ~8 GB/s unidirezionale)
    Dev (pageable)   : ~45 ms  (overhead DMA copy host-side)
    Target streaming : < 2 ms first-layer-visible con pipeline
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class GPUTransfer:
    """Gestisce trasferimenti DDR4 ↔ VRAM su RTX 3090 (device 0).

    Args:
        device_id: CUDA device index (0 = RTX 3090 in setup corrente).
    """

    def __init__(self, device_id: int = 0) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch non disponibile — GPU transfer non funzionale.")
        self._device = torch.device(f"cuda:{device_id}")
        self._device_id = device_id

    # ── host → device (promozione DDR4 → VRAM) ────────────────────────────────

    def to_vram(self, data: np.ndarray,
                stream: torch.cuda.Stream | None = None) -> torch.Tensor:
        """Trasferisce uno shard da DDR4 a VRAM.

        NOTE: senza pinned memory, il driver CUDA esegue una copia intermedia
        su host-pinned buffer interno. Overhead misurabile vs. cudaMallocHost.

        Args:
            data:   numpy array uint8 (shard in DDR4).
            stream: CUDA stream per overlap asincrono (None = stream default).

        Returns:
            torch.Tensor su GPU (dtype uint8).
        """
        if stream is not None:
            with torch.cuda.stream(stream):
                return torch.from_numpy(data).to(self._device, non_blocking=True)
        return torch.from_numpy(data).to(self._device)

    # ── device → host (eviction VRAM → DDR4) ──────────────────────────────────

    def to_ddr4(self, tensor: torch.Tensor) -> np.ndarray:
        """Trasferisce uno shard da VRAM a DDR4 (eviction write-back).

        Args:
            tensor: torch.Tensor su GPU.

        Returns:
            numpy array uint8 in DDR4.
        """
        return tensor.detach().cpu().numpy()

    # ── utils ──────────────────────────────────────────────────────────────────

    def vram_free_bytes(self) -> int:
        """VRAM libera corrente su device (bytes)."""
        free, _total = torch.cuda.mem_get_info(self._device_id)
        return free

    def vram_total_bytes(self) -> int:
        """VRAM totale su device (bytes). Atteso: 24 GB per RTX 3090."""
        _free, total = torch.cuda.mem_get_info(self._device_id)
        return total

    def create_stream(self) -> torch.cuda.Stream:
        """Crea un nuovo CUDA stream per trasferimenti asincroni."""
        return torch.cuda.Stream(device=self._device)

    def empty_cache(self) -> None:
        """Rilascia al driver CUDA i blocchi liberati dal caching allocator
        di PyTorch.

        torch.cuda.mem_get_info() (usato da vram_free_bytes()) riporta la
        memoria libera a livello driver, non quella già liberata da un
        `del tensor` ma ancora trattenuta in cache dall'allocator di
        PyTorch per riuso. Senza questa chiamata dopo un'eviction,
        vram_free_bytes() resta invariato e un ciclo evict_to_free_vram()
        che si affida a quel numero per decidere quando fermarsi continua
        a evictare oltre il necessario — bug reale, trovato eseguendo i
        test su hardware reale (mai riprodotto in nessun mock/CPU test).
        """
        torch.cuda.empty_cache()
