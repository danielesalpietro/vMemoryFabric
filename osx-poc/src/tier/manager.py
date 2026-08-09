"""M2 — EMH Tier Manager — orchestratore principale.

Coordina promozione/evizione shard tra i tier EMH disponibili in dev:
    NVMe (EMH-3) → DDR4 (EMH-1c) → VRAM RTX 3090 (EMH-1a)

Interfacce verso altri moduli:
    ← EAT  (M1): lettura tier corrente, aggiornamento post-transizione
    → ES   (M3): notifica eviction_candidates[], ricezione prefetch_queue[]
    → Metrics: esposizione latenza promozione, hit rate, tier distribution
"""
from __future__ import annotations

import logging

from eat.eat import ExpertAccessTable
from eat.types import ExpertID, ShardID, Tier

from .gpu import GPUTransfer
from .io import AsyncNVMeIO
from .policies import EvictionCandidate, LRUPolicy, SEEPolicy

log = logging.getLogger(__name__)


class TierManager:
    """Gestisce il ciclo di vita degli shard attraverso i tier EMH.

    Args:
        eat:        Riferimento alla Expert Access Table (M1).
        nvme_path:  Path del volume NVMe cold storage.
        gpu_device: CUDA device ID (0 = RTX 3090).
        use_see:    Se True usa SEE policy; altrimenti LRU puro.
    """

    def __init__(
        self,
        eat: ExpertAccessTable,
        nvme_path: str = "/data/nvme",
        gpu_device: int = 0,
        use_see: bool = True,
    ) -> None:
        self._eat    = eat
        self._io     = AsyncNVMeIO(base_path=nvme_path)
        self._gpu    = GPUTransfer(device_id=gpu_device)
        self._policy = SEEPolicy() if use_see else LRUPolicy()

    # ── promotions ─────────────────────────────────────────────────────────────

    async def promote(self, expert_id: ExpertID, shard_idx: ShardID,
                      target_tier: Tier) -> float:
        """Promuove uno shard verso un tier superiore.

        Percorso supportato in dev: NVME → DDR4 → VRAM.
        PMEM (tra DDR4 e NVME) sarà inserito quando disponibile.

        Args:
            expert_id:   ID expert.
            shard_idx:   Indice shard.
            target_tier: Tier di destinazione.

        Returns:
            Latenza della transizione in secondi.

        Raises:
            ValueError: tier di destinazione non raggiungibile dal tier corrente.
            MemoryError: Slab Allocator o VRAM esauriti.
        """
        raise NotImplementedError("TODO Sprint 2")

    async def _nvme_to_ddr4(self, expert_id: ExpertID, shard_idx: ShardID) -> float:
        """NVMe → DDR4: asyncio + aiofiles (proxy io_uring).

        Returns: latenza in secondi.
        """
        raise NotImplementedError("TODO Sprint 2")

    async def _ddr4_to_vram(self, expert_id: ExpertID, shard_idx: ShardID) -> float:
        """DDR4 → VRAM: cudaMemcpy standard (no pinned — dev constraint).

        Returns: latenza in secondi.
        """
        raise NotImplementedError("TODO Sprint 2")

    # ── evictions ──────────────────────────────────────────────────────────────

    async def evict(self, expert_id: ExpertID, shard_idx: ShardID) -> None:
        """Eviction manuale di uno shard (verso tier inferiore).

        Scrive lo shard nel tier inferiore, poi aggiorna EAT.
        """
        raise NotImplementedError("TODO Sprint 2")

    async def evict_to_free_vram(self, target_free_bytes: int,
                                 context_vec: list[float] | None = None) -> list[EvictionCandidate]:
        """Evict automatico da VRAM fino a liberare target_free_bytes.

        Usa SEE policy (o LRU fallback) per selezionare i candidati.
        Session-scoped: eviction cross-sessione bloccata durante sessione attiva.

        Args:
            target_free_bytes: Quanta VRAM liberare.
            context_vec:       Vettore semantico PT-PEP per SEE (None = LRU).

        Returns:
            Lista di shard evicted.
        """
        raise NotImplementedError("TODO Sprint 2")

    # ── prefetch ───────────────────────────────────────────────────────────────

    async def prefetch(self, prefetch_queue: list[tuple[ExpertID, ShardID]]) -> None:
        """Prefetch asincrono di una lista di shard verso VRAM.

        Chiamato da Expert Scheduler (M3) con la prefetch_queue PT-PEP.
        """
        raise NotImplementedError("TODO Sprint 2 — integrazione M3 in Sprint 3")

    # ── stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Metriche per Prometheus: latenza per tier, hit rate, VRAM free."""
        raise NotImplementedError("TODO Sprint 2")
