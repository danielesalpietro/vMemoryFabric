"""M2 — EMH Tier Manager — orchestratore principale.

Coordina promozione/evizione shard tra i tier EMH disponibili:
    NVMe (EMH-3) → DDR4 (EMH-1c) → VRAM RTX 3090 (EMH-1a)   [sempre]
    NVMe (EMH-3) → PMEM (EMH-2) → DDR4 (EMH-1c)              [solo se
        pmem_path impostato — host con mount DAX reale, vedi pmem.py]

Interfacce verso altri moduli:
    ← EAT  (M1): lettura tier corrente, aggiornamento post-transizione
    → ES   (M3): notifica eviction_candidates[], ricezione prefetch_queue[]
    → Metrics: esposizione latenza promozione, hit rate, tier distribution
"""
from __future__ import annotations

import asyncio
import logging
import time

from eat.eat import ExpertAccessTable
from eat.types import ExpertID, ShardID, Tier

from .gpu import GPUTransfer
from .io import AsyncNVMeIO
from .pmem import PMEMTransfer
from .policies import EvictionCandidate, LRUPolicy, SEEPolicy

log = logging.getLogger(__name__)

_Key = tuple[ExpertID, ShardID]


class TierManager:
    """Gestisce il ciclo di vita degli shard attraverso i tier EMH.

    Args:
        eat:        Riferimento alla Expert Access Table (M1).
        nvme_path:  Path del volume NVMe cold storage.
        gpu_device: CUDA device ID (0 = RTX 3090).
        use_see:    Se True usa SEE policy; altrimenti LRU puro.
        pmem_path:  Path del mount DAX per EMH-2 (es. /data/pmem nel
                    container). None (default) = tier PMEM disabilitato,
                    stesso comportamento di prima di questo parametro —
                    la maggior parte degli ambienti dev (Docker-on-
                    Windows/WSL2, RunPod) non ha PMEM. Vedi tier/pmem.py
                    e osx-poc/LOGBOOK_NEW_Z8.md "passo 4".
        pmem_n_slots: Slot pre-allocati nel pool PMEM, solo se pmem_path
                    è impostato. Default piccolo (dev/bench), non
                    dimensionato per un pool di produzione.
    """

    def __init__(
        self,
        eat: ExpertAccessTable,
        nvme_path: str = "/data/nvme",
        gpu_device: int = 0,
        use_see: bool = True,
        pmem_path: str | None = None,
        pmem_n_slots: int = 4,
    ) -> None:
        self._eat    = eat
        self._io     = AsyncNVMeIO(base_path=nvme_path)
        self._gpu    = GPUTransfer(device_id=gpu_device)
        self._policy = SEEPolicy() if use_see else LRUPolicy()
        # Bookkeeping di proprietà del TierManager (non di EATEntry — vedi
        # decisione Sprint 2): quale slot slab DDR4 possiede quale shard,
        # e quale torch.Tensor VRAM possiede quale shard.
        self._slots: dict[_Key, int] = {}
        self._vram: dict[_Key, "torch.Tensor"] = {}
        # EMH-2 (PMEM): stesso pattern di self._slots, ma per gli slot del
        # pool PMEM. self._pmem è None se il tier non è disponibile su
        # questo host — ogni hop NVME<->PMEM/PMEM<->DDR4 verifica esplicito.
        self._pmem: PMEMTransfer | None = None
        self._pmem_slots: dict[_Key, int] = {}
        if pmem_path is not None:
            self._pmem = PMEMTransfer(mount_path=pmem_path, n_slots=pmem_n_slots)
            self._pmem.initialize()
        # Lock per-key: serializza l'intera transizione di tier per uno
        # stesso shard. Senza questo, due promote() concorrenti sulla
        # stessa chiave vedrebbero entrambi lo stesso tier di partenza
        # prima dell'await sull'I/O, allocherebbero due slot slab distinti
        # per un solo shard logico — uno slab leak garantito sotto asyncio,
        # anche a thread singolo, non un edge case teorico.
        self._locks: dict[_Key, asyncio.Lock] = {}

    @property
    def eat(self) -> ExpertAccessTable:
        """Expert Access Table sottostante — esposta per chi (es.
        scheduler.gcsg.GCSGWorker, M3) deve leggere/scrivere hotness reale
        al di fuori del solo ciclo promote()/evict() (2026-08-12, issue
        #17). Stesso pattern di ExpertAccessTable.slab, esposto per lo
        stesso motivo: proprietà del componente sottostante, non una
        copia scollegata."""
        return self._eat

    @property
    def pmem(self) -> PMEMTransfer | None:
        """Pool PMEM (EMH-2) sottostante, o None se questo host non ha
        pmem_path impostato — esposta per lo shutdown esplicito da parte
        del chiamante (es. benchmark), stesso pattern di `eat` sopra."""
        return self._pmem

    def _lock_for(self, key: _Key) -> asyncio.Lock:
        """Lock asyncio per-key, creato lazy alla prima transizione su quella key."""
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    # ── promotions ─────────────────────────────────────────────────────────────

    async def promote(self, expert_id: ExpertID, shard_idx: ShardID,
                      target_tier: Tier) -> float:
        """Promuove uno shard verso un tier superiore.

        Percorso base: NVME → DDR4 → VRAM (sempre disponibile).
        Percorso EMH-2 (solo se self._pmem è impostato — vedi __init__):
        NVME → PMEM (single-hop) e PMEM → DDR4 (single-hop). Non ancora
        incatenato in un percorso NVME → PMEM → VRAM a due hop come
        NVME → VRAM lo è per DDR4/VRAM sotto — il chiamante che vuole
        PMEM come tappa intermedia verso VRAM chiama promote() due volte
        (NVME→PMEM, poi PMEM→DDR4, poi DDR4→VRAM). NVME → DDR4 diretto
        resta invariato e disponibile a prescindere da self._pmem: PMEM
        è una rotta aggiuntiva, non una sostituzione.

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
        key = (expert_id, shard_idx)
        async with self._lock_for(key):
            entry = self._eat.lookup(expert_id, shard_idx)
            if entry is None:
                raise ValueError(f"shard non presente in EAT: {key}")
            current = entry.tier

            if current == Tier.NVME and target_tier == Tier.DDR4:
                return await self._nvme_to_ddr4(expert_id, shard_idx)
            if current == Tier.DDR4 and target_tier == Tier.VRAM:
                return await self._ddr4_to_vram(expert_id, shard_idx)
            if current == Tier.NVME and target_tier == Tier.VRAM:
                # Unico caso a due hop: incatena NVME→DDR4 poi DDR4→VRAM
                # sotto lo stesso lock — nessun altro promote()/evict()
                # sulla stessa key può intercalarsi a metà catena.
                t1 = await self._nvme_to_ddr4(expert_id, shard_idx)
                t2 = await self._ddr4_to_vram(expert_id, shard_idx)
                return t1 + t2
            if current == Tier.NVME and target_tier == Tier.PMEM:
                if self._pmem is None:
                    raise ValueError("tier PMEM non disponibile su questo host (pmem_path non impostato)")
                return await self._nvme_to_pmem(expert_id, shard_idx)
            if current == Tier.PMEM and target_tier == Tier.DDR4:
                return await self._pmem_to_ddr4(expert_id, shard_idx)
            raise ValueError(
                f"promote non supportata: {current.name} -> {target_tier.name}"
            )

    async def _nvme_to_ddr4(self, expert_id: ExpertID, shard_idx: ShardID) -> float:
        """NVMe → DDR4: asyncio + aiofiles (proxy io_uring).

        Chiamata con il lock della key già posseduto da promote().

        Returns: latenza in secondi.
        """
        key = (expert_id, shard_idx)
        t0 = time.monotonic()
        data = await self._io.read_shard(expert_id, shard_idx)
        slot_idx = self._eat.slab.alloc(expert_id, shard_idx, len(data))
        buffer = self._eat.slab.get_buffer(slot_idx)
        buffer[: len(data)] = data
        self._slots[key] = slot_idx
        self._eat.update_tier(expert_id, shard_idx, Tier.DDR4)
        return time.monotonic() - t0

    async def _nvme_to_pmem(self, expert_id: ExpertID, shard_idx: ShardID) -> float:
        """NVMe → PMEM (EMH-2): stessa lettura di _nvme_to_ddr4(), scrive
        nel pool PMEM (mmap DAX) invece che nel pool DDR4 (SlabAllocator).

        Chiamata con il lock della key già posseduto da promote(), e solo
        se self._pmem è impostato (verificato dal chiamante).

        Returns: latenza in secondi.
        """
        key = (expert_id, shard_idx)
        t0 = time.monotonic()
        data = await self._io.read_shard(expert_id, shard_idx)
        slot_idx = self._pmem.alloc(expert_id, shard_idx, len(data))
        self._pmem.write(slot_idx, data)
        self._pmem_slots[key] = slot_idx
        self._eat.update_tier(expert_id, shard_idx, Tier.PMEM)
        return time.monotonic() - t0

    async def _pmem_to_ddr4(self, expert_id: ExpertID, shard_idx: ShardID) -> float:
        """PMEM → DDR4: copia dal pool PMEM (mmap DAX, zero-copy read) al
        pool DDR4 (SlabAllocator) — un vero memcpy, non I/O asincrono
        come l'hop NVMe→DDR4 (niente da awaitare qui, la mmap è già
        mappata in process; la funzione resta `async def` solo per
        coerenza di firma con gli altri hop, non perché ceda il
        controllo internamente).

        Chiamata con il lock della key già posseduto da promote().

        Returns: latenza in secondi.
        """
        key = (expert_id, shard_idx)
        t0 = time.monotonic()
        pmem_slot_idx = self._pmem_slots[key]
        data = self._pmem.read(pmem_slot_idx)
        ddr4_slot_idx = self._eat.slab.alloc(expert_id, shard_idx, len(data))
        buffer = self._eat.slab.get_buffer(ddr4_slot_idx)
        buffer[: len(data)] = data
        self._pmem.free(pmem_slot_idx)
        del self._pmem_slots[key]
        self._slots[key] = ddr4_slot_idx
        self._eat.update_tier(expert_id, shard_idx, Tier.DDR4)
        return time.monotonic() - t0

    async def _ddr4_to_vram(self, expert_id: ExpertID, shard_idx: ShardID) -> float:
        """DDR4 → VRAM: cudaMemcpy standard (no pinned — dev constraint).

        Chiamata con il lock della key già posseduto da promote().

        Se self._gpu.to_vram() solleva (es. CUDA OOM), lo fa PRIMA che lo
        slot slab venga liberato o l'EAT aggiornata — lo shard resta
        pienamente valido in DDR4 (slot ancora posseduto, tier ancora
        DDR4), non in uno stato a metà promozione. Nessun retry/recovery
        attivo in M2: l'eccezione si propaga al chiamante a stato intatto.

        Returns: latenza in secondi.
        """
        key = (expert_id, shard_idx)
        t0 = time.monotonic()
        slot_idx = self._slots[key]
        buffer = self._eat.slab.get_buffer(slot_idx)
        tensor = self._gpu.to_vram(buffer)
        self._vram[key] = tensor
        self._eat.slab.free(slot_idx)
        del self._slots[key]
        self._eat.update_tier(expert_id, shard_idx, Tier.VRAM)
        return time.monotonic() - t0

    # ── live-tensor promotion (M3 shadow pool, issue #17) ─────────────────────

    async def promote_live_tensor(
        self, expert_id: ExpertID, shard_idx: ShardID,
        cpu_data, pin: bool = False,
    ) -> "torch.Tensor":
        """Promuove in VRAM un tensore CPU già in memoria di processo — non
        uno shard su file NVMe (2026-08-12, issue #17).

        Diverso da promote(): quello porta uno shard attraverso l'intera
        catena NVMe→DDR4→VRAM leggendo da AsyncNVMeIO. Questo serve asset
        che vivono già lato host per costruzione — es. GCSGWorker (M3), i
        cui shadow expert sono slice/parametri del modello vLLM già
        caricato (residenti in GPU o offloaded su CPU da vLLM stesso), mai
        file separati sul volume NVMe. La entry EAT viene comunque creata
        (a Tier.DDR4, se assente — quella è la tier di partenza onesta per
        questi shard) e aggiornata a Tier.VRAM dopo il transfer, così
        EAT/TierManager restano la fonte di verità sul tier reale anche
        per questi asset, bypassando solo l'I/O NVMe che non serve qui.

        Idempotente: se lo shard è già in VRAM, ritorna il tensore
        esistente senza ripetere il transfer.

        Args:
            expert_id: ID expert (vedi convenzione GCSG: un "expert" è
                       l'FFN con quell'indice su un layer specifico —
                       shard_idx qui è tipicamente il layer_id, non un
                       chunk-in-byte come nel flusso NVMe).
            shard_idx: Indice shard/layer.
            cpu_data:  numpy.ndarray o torch.Tensor CPU-resident.
            pin:       Passato a GPUTransfer.to_vram() — vedi la sua
                       docstring per lo stato di verifica.

        Returns:
            torch.Tensor su GPU.

        Raises:
            ValueError: entry EAT esistente ma non in DDR4/VRAM (stato
                        inatteso per un asset live — non gestito qui).
        """
        key = (expert_id, shard_idx)
        async with self._lock_for(key):
            entry = self._eat.lookup(expert_id, shard_idx)
            if entry is not None and entry.tier == Tier.VRAM:
                return self._vram[key]

            if entry is None:
                nbytes = getattr(cpu_data, "nbytes", None)
                if nbytes is None:   # torch.Tensor non ha sempre .nbytes su versioni vecchie
                    nbytes = cpu_data.element_size() * cpu_data.nelement()
                self._eat.insert(expert_id, shard_idx, tier=Tier.DDR4, size_bytes=nbytes)
            elif entry.tier != Tier.DDR4:
                raise ValueError(
                    f"promote_live_tensor: {key} è in tier {entry.tier.name}, "
                    "atteso DDR4 o assente."
                )

            t0 = time.monotonic()
            tensor = self._gpu.to_vram(cpu_data, pin=pin)
            self._vram[key] = tensor
            self._eat.update_tier(expert_id, shard_idx, Tier.VRAM)
            log.debug(
                "TierManager: promote_live_tensor %s -> VRAM in %.1f ms (pin=%s)",
                key, (time.monotonic() - t0) * 1000, pin,
            )
            return tensor

    # ── evictions ──────────────────────────────────────────────────────────────

    async def evict(self, expert_id: ExpertID, shard_idx: ShardID) -> None:
        """Eviction manuale di uno shard (verso tier inferiore).

        Scrive lo shard nel tier inferiore, poi aggiorna EAT (chiama
        eat.update_tier(), NON eat.evict() — quest'ultimo è il metodo M1
        che rimuove del tutto la riga dalla EAT, un'operazione diversa
        che condivide solo il nome).

        Single-hop soltanto: a differenza di promote(), che incatena
        NVME→VRAM in due hop, evict(VRAM→NVME) in un'unica chiamata
        solleva ValueError. L'asimmetria è intenzionale, non un'omissione.

        Raises:
            ValueError: shard non presente in EAT, o tier corrente non
                        VRAM/DDR4 (nessun hop disponibile o già in NVME).
        """
        key = (expert_id, shard_idx)
        async with self._lock_for(key):
            entry = self._eat.lookup(expert_id, shard_idx)
            if entry is None:
                raise ValueError(f"shard non presente in EAT: {key}")
            current = entry.tier

            if current == Tier.VRAM:
                tensor = self._vram.pop(key)
                data = self._gpu.to_ddr4(tensor)
                slot_idx = self._eat.slab.alloc(expert_id, shard_idx, data.nbytes)
                buffer = self._eat.slab.get_buffer(slot_idx)
                buffer[: len(data)] = data
                self._slots[key] = slot_idx
                # `tensor` è ancora una live reference qui — self._vram.pop()
                # toglie solo il riferimento del dict. torch.cuda.empty_cache()
                # può restituire al driver solo i blocchi SENZA reference vive,
                # quindi il `del` esplicito è necessario, non solo cosmetico:
                # senza, empty_cache() è un no-op silenzioso e vram_free_bytes()
                # resta invariato (bug reale trovato su hardware reale — vedi
                # GPUTransfer.empty_cache()).
                del tensor
                self._gpu.empty_cache()
                self._eat.update_tier(expert_id, shard_idx, Tier.DDR4)
                return

            if current == Tier.DDR4:
                slot_idx = self._slots[key]
                buffer = self._eat.slab.get_buffer(slot_idx)
                await self._io.write_shard(expert_id, shard_idx, buffer)
                self._eat.slab.free(slot_idx)
                del self._slots[key]
                self._eat.update_tier(expert_id, shard_idx, Tier.NVME)
                return

            if current == Tier.PMEM:
                if self._pmem is None:
                    raise ValueError("tier PMEM non disponibile su questo host (pmem_path non impostato)")
                slot_idx = self._pmem_slots[key]
                buffer = self._pmem.read(slot_idx)
                await self._io.write_shard(expert_id, shard_idx, buffer)
                self._pmem.free(slot_idx)
                del self._pmem_slots[key]
                self._eat.update_tier(expert_id, shard_idx, Tier.NVME)
                return

            raise ValueError(
                f"evict non supportata da tier {current.name} "
                "(solo VRAM->DDR4, DDR4->NVME o PMEM->NVME, single-hop)"
            )

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

        Raises:
            MemoryError: il tier VRAM è vuoto e target_free_bytes non è
                         ancora raggiunto.
        """
        evicted: list[EvictionCandidate] = []
        while self._gpu.vram_free_bytes() < target_free_bytes:
            candidates = self._eat.get_tier(Tier.VRAM)
            if not candidates:
                raise MemoryError(
                    f"impossibile liberare {target_free_bytes} byte di VRAM: "
                    "tier VRAM vuoto"
                )
            if isinstance(self._policy, SEEPolicy):
                ranked = self._policy.rank(candidates, n=1, context_vec=context_vec)
            else:
                ranked = self._policy.rank(candidates, n=1)
            victim = ranked[0]
            await self.evict(victim.entry.expert_id, victim.entry.shard_idx)
            evicted.append(victim)
        return evicted

    # ── prefetch ───────────────────────────────────────────────────────────────

    async def prefetch(self, prefetch_queue: list[tuple[ExpertID, ShardID]]) -> None:
        """Prefetch asincrono di una lista di shard verso VRAM.

        Chiamato da Expert Scheduler (M3) con la prefetch_queue PT-PEP.
        Le promote() girano concorrentemente (asyncio.gather) — è il punto
        di un prefetch asincrono — e un fallimento su un singolo shard
        (es. MemoryError per VRAM piena) viene loggato, non propagato,
        così non abortisce l'intero batch.
        """
        results = await asyncio.gather(
            *(self.promote(expert_id, shard_idx, Tier.VRAM)
              for expert_id, shard_idx in prefetch_queue),
            return_exceptions=True,
        )
        for (expert_id, shard_idx), result in zip(prefetch_queue, results):
            if isinstance(result, Exception):
                log.warning(
                    "prefetch fallito per (expert_id=%s, shard_idx=%s): %s",
                    expert_id, shard_idx, result,
                )

    # ── stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Metriche per Prometheus: latenza per tier, hit rate, VRAM free."""
        stats = self._eat.stats()
        stats.update({
            "vram_free_bytes": self._gpu.vram_free_bytes(),
            "vram_total_bytes": self._gpu.vram_total_bytes(),
            "ddr4_slots_free": self._eat.slab.free_slots,
            "ddr4_slots_used": self._eat.slab.used_slots,
            "policy": type(self._policy).__name__,
        })
        return stats
