"""M2 — GPU Transfer (DDR4 → VRAM RTX 3090).

Adattamenti rispetto a spec OSX v1.0:
    - Pinned memory (cudaMallocHost) NON disponibile in Docker su Windows
      (WSL2) — default storico di questo modulo: to_vram() usa cudaMemcpy
      standard con pageable host memory a meno di pin=True esplicito.
    - Single GPU: solo RTX 3090 (device 0). Dual-GPU deferred.
    - Nessuna CUDA stream pipeline per ora — sarà aggiunta in Sprint 2
      per misurare l'overlap compute/transfer.

    **Aggiornamento (2026-08-12, Sprint 4 sotto-obiettivo 2):** pinning
    reale (torch.Tensor.pin_memory()) verificato sicuro e stabile sotto
    carico sostenuto su Linux reale (RunPod, non WSL2) — 1000 cicli
    pin-allocate/H2D/D2H/confronto-byte, 0 mismatch, nessun degrado (vedi
    LOGBOOK.md, GCSG report §9). to_vram() ora accetta pin=True per
    usarlo davvero — default resta False (comportamento storico
    invariato, il chiamante decide in base alla piattaforma reale, es.
    via vllm.platforms.interface.in_wsl() quando disponibile — vedi
    scheduler.gcsg.GCSGWorker._should_pin_transfers()). Non ancora
    validato SPECIFICAMENTE per questo path (era stato validato lo
    shard-transfer generico, non ancora un run reale attraverso
    TierManager.promote_live_tensor() con pin=True) — primo item da
    verificare sul pod.

Target latenza DDR4 → VRAM (256 MB shard):
    Teorico pinned   : ~32 ms  (PCIe Gen3 ~8 GB/s unidirezionale)
    Dev (pageable)   : ~45 ms  (overhead DMA copy host-side)
    Target streaming : < 2 ms first-layer-visible con pipeline
"""
from __future__ import annotations
import logging
from typing import Optional, Union
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

log = logging.getLogger(__name__)


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

    def to_vram(self, data: Union[np.ndarray, "torch.Tensor"],
                stream: Optional["torch.cuda.Stream"] = None,
                pin: bool = False) -> "torch.Tensor":
        """Trasferisce uno shard da DDR4 a VRAM.

        NOTE: senza pinned memory (pin=False, default), il driver CUDA
        esegue una copia intermedia su host-pinned buffer interno.
        Overhead misurabile vs. cudaMallocHost — vedi pin=True sotto.

        Args:
            data:   numpy array uint8 (shard in DDR4) oppure un
                    torch.Tensor già esistente (CPU o GPU — usato da
                    scheduler.gcsg.GCSGWorker, i cui shadow expert sono
                    tensori reali già del modello caricato, non byte
                    grezzi da un file NVMe).
            stream: CUDA stream per overlap asincrono (None = stream
                    default). Se fornito, il transfer resta non_blocking
                    a prescindere da pin — comportamento storico invariato.
            pin:    Se True, tenta un pin_memory() reale prima del
                    transfer (verificato sicuro su Linux reale, non
                    ancora specificamente su questo path — vedi docstring
                    di modulo). Fallback silenzioso a pageable se
                    pin_memory() solleva. Default False: comportamento
                    identico a prima di questa opzione.

        Returns:
            torch.Tensor su GPU.
        """
        if isinstance(data, np.ndarray):
            base = torch.from_numpy(data)
        else:
            base = data.detach()
            if base.device.type != "cpu":
                base = base.cpu()

        if pin:
            try:
                base = base.pin_memory()
            except Exception as e:
                log.warning("GPUTransfer: pin_memory() fallito (%s) — fallback a pageable.", e)
                pin = False

        if stream is not None:
            with torch.cuda.stream(stream):
                return base.to(self._device, non_blocking=True)
        return base.to(self._device, non_blocking=pin)

    # ── device → host (eviction VRAM → DDR4) ──────────────────────────────────

    def to_ddr4(self, tensor: "torch.Tensor") -> np.ndarray:
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

    def create_stream(self) -> "torch.cuda.Stream":
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
