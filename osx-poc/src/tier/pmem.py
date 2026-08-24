"""M2 — PMEM Transfer (EMH-2, Optane DC Persistent Memory, fsdax).

Issue #33/#45 follow-up (see osx-poc/LOGBOOK_NEW_Z8.md, "passo 4"): the
Z8 G4 bare-metal host has two 252 GiB Optane DIMM regions. Region0 hosts
the OS (sector mode, unrelated to this module). Region1 was reconfigured
via `ndctl` from `sector` to `fsdax` mode (`/dev/pmem1`, byte-addressable),
then formatted XFS and mounted with `mount -o dax` (`dax=always`) at
`/mnt/pmem_emh2` on the host — bind-mounted into the dev container at
`/data/pmem` via `docker-compose.override.yml` (gitignored, host-specific;
see `docker-compose.override.yml.example`).

Design: a fixed-slot pool file on that DAX-mounted filesystem,
memory-mapped with `numpy.memmap` — same fixed-slot-pool shape as
`eat.slab.SlabAllocator` (DDR4, anonymous `np.empty`), but backed by a
real file on PMEM instead of anonymous DRAM. The pool file is
`posix_fallocate`-d to its full size before mapping specifically to avoid
a sparse file: an unallocated extent would still fault in additional
blocks on first write, defeating the point of measuring real PMEM-backed
access from the first read.

Known limitation, stated plainly rather than implied: this uses a normal
mmap (`numpy.memmap`), not `MAP_SYNC` — writes go through the normal
page-cache-coherent DAX path, which gives byte-addressable access and
avoids double-buffering through the page cache, but does NOT give the
fsync-free durability guarantee `MAP_SYNC` provides on a real fsdax mount.
Getting an actual `MAP_SYNC` mapping needs a raw `mmap.mmap()` call with
`MAP_SYNC | MAP_SHARED_VALIDATE` (not exposed by Python's `mmap` module
today) — out of scope for this pass, which is about characterizing
promotion latency and raw throughput for this tier, not write durability
semantics. `flush()` below calls `.flush()` on the memmap (msync), which
is the honest level of durability this module actually provides.

Target latency PMEM ↔ DDR4 (256 MB shard):
    Optane DC Gen1/2 sequential read : ~2-3 GB/s (order of magnitude
        below DDR4-to-DDR4 copy bandwidth measured on this host, ~20-36
        GB/s — see perf_test_hardware.py runs in LOGBOOK_NEW_Z8.md) but
        far above NVMe Gen3 (~3.5 GB/s) at far higher capacity (252 GiB
        vs 256 GB DDR4 on this host) and persistent across reboot.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eat.types import SHARD_SIZE_BYTES

log = logging.getLogger(__name__)


@dataclass
class _PMEMSlotMetadata:
    expert_id: int
    shard_idx: int
    size_bytes: int  # effettivo (<= shard_size del pool)


class PMEMTransfer:
    """Pool allocator a slot fissi per shard neurali, backed da PMEM
    (mount DAX) invece che DRAM anonima.

    Args:
        mount_path:  Directory radice del mount DAX (es. /data/pmem nel
                     container, bind-mount di /mnt/pmem_emh2 sull'host).
        n_slots:     Numero di slot pre-allocati nel pool.
        shard_size:  Dimensione fissa per slot (bytes). Default:
                     SHARD_SIZE_BYTES (256 MB), stessa costante di
                     SlabAllocator — un pool PMEM più piccolo del DDR4
                     equivalente non avrebbe senso allo stesso
                     shard_size.
        pool_filename: Nome del file pool sul mount DAX.
    """

    def __init__(
        self,
        mount_path: str,
        n_slots: int,
        shard_size: int = SHARD_SIZE_BYTES,
        pool_filename: str = "emh2_pool.bin",
    ) -> None:
        self._mount_path = Path(mount_path)
        self._shard_size = shard_size
        self._n_slots = n_slots
        self._pool_path = self._mount_path / pool_filename
        self._free_slots: list[int] = list(range(n_slots))
        self._alloc_map: dict[int, _PMEMSlotMetadata] = {}
        self._mmap: np.memmap | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Crea (se assente) e mappa il file pool sul mount DAX.

        `posix_fallocate` forza l'allocazione reale di tutti i blocchi
        PRIMA della mmap — senza questo, np.memmap crea un file sparse
        (truncate a dimensione target senza scrivere), e i buchi vengono
        allocati lazy al primo write, il che confonderebbe qualunque
        benchmark di throughput con il costo one-off dell'allocazione
        blocchi invece del solo costo di scrittura PMEM.
        """
        self._mount_path.mkdir(parents=True, exist_ok=True)
        total_bytes = self._n_slots * self._shard_size

        is_new = not self._pool_path.exists() or self._pool_path.stat().st_size != total_bytes
        if is_new:
            fd = os.open(self._pool_path, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                os.posix_fallocate(fd, 0, total_bytes)
            finally:
                os.close(fd)

        self._mmap = np.memmap(
            self._pool_path, dtype=np.uint8, mode="r+",
            shape=(self._n_slots, self._shard_size),
        )

    def shutdown(self) -> None:
        """Flush (msync) e rilascia la mappatura. Non cancella il file
        pool su PMEM — persiste per design (a differenza del pool DDR4
        di SlabAllocator, che è DRAM anonima e sparisce comunque)."""
        if self._mmap is not None:
            self._mmap.flush()
            self._mmap = None
        self._free_slots = list(range(self._n_slots))
        self._alloc_map = {}

    def flush(self) -> None:
        """msync esplicito senza rilasciare la mappatura — vedi limitazione
        MAP_SYNC nel docstring di modulo: questo è il livello di durabilità
        reale offerto (page-cache flush), non un ordering barrier hardware."""
        if self._mmap is not None:
            self._mmap.flush()

    # ── alloc / free ───────────────────────────────────────────────────────────

    def alloc(self, expert_id: int, shard_idx: int, size_bytes: int) -> int:
        """Alloca uno slot dal pool. Stessa semantica di
        SlabAllocator.alloc() (osx-poc/src/eat/slab.py) — vedi lì per il
        perché su size_bytes/is_tail non replicato qui (PMEM non ha
        ancora un consumer variable-tail)."""
        if self._mmap is None:
            raise RuntimeError("PMEMTransfer non inizializzato — chiamare initialize()")
        if not (0 <= size_bytes <= self._shard_size):
            raise ValueError(f"size_bytes {size_bytes} fuori range [0, {self._shard_size}]")
        if not self._free_slots:
            raise MemoryError("pool PMEM esaurito")
        slot_idx = self._free_slots.pop()
        self._alloc_map[slot_idx] = _PMEMSlotMetadata(expert_id, shard_idx, size_bytes)
        return slot_idx

    def free(self, slot_idx: int) -> None:
        if slot_idx not in self._alloc_map:
            raise KeyError(f"slot {slot_idx} non allocato")
        del self._alloc_map[slot_idx]
        self._free_slots.append(slot_idx)

    # ── read / write ───────────────────────────────────────────────────────────
    # Sincroni, non async: a differenza di AsyncNVMeIO (file I/O via
    # aiofiles/io_uring), l'accesso a una regione mmap-ata è già un
    # memcpy diretto, non una syscall bloccante da wrappare in un
    # executor — wrapparlo in `async def` senza un vero `await` interno
    # non comprerebbe nulla, solo un livello di indirection in più.

    def write(self, slot_idx: int, data: np.ndarray) -> None:
        """Scrive `data` nello slot (memcpy diretto sulla regione mmap-ata)."""
        if self._mmap is None:
            raise RuntimeError("PMEMTransfer non inizializzato — chiamare initialize()")
        self._mmap[slot_idx][: len(data)] = data

    def read(self, slot_idx: int) -> np.ndarray:
        """Ritorna una VIEW (zero-copy) sullo slot, tagliata alla
        dimensione REALE scritta in alloc() (`size_bytes`), non l'intero
        slot fisso da `shard_size` — a differenza di
        SlabAllocator.get_buffer() (eat/slab.py), che ritorna sempre lo
        slot intero indipendentemente dal payload reale. Quella
        differenza di comportamento è esattamente la causa radice di
        issue #48 (bench_ddr4_to_vram trasferiva 256 MB fissi invece dei
        4 MB sintetici dichiarati) — qui si evita lo stesso bug fin
        dall'inizio invece di ripeterlo: un chiamante che fa `len(data)`
        sul risultato di read() ottiene la dimensione reale, non quella
        dello slot.

        Il chiamante che deve sopravvivere oltre un free()/shutdown() di
        questo slot deve copiare esplicitamente (`.copy()`) — resta una
        view, non una copia."""
        if self._mmap is None:
            raise RuntimeError("PMEMTransfer non inizializzato — chiamare initialize()")
        meta = self._alloc_map.get(slot_idx)
        if meta is None:
            raise KeyError(f"slot {slot_idx} non allocato")
        return self._mmap[slot_idx][: meta.size_bytes]

    # ── stats ──────────────────────────────────────────────────────────────────

    @property
    def free_slots(self) -> int:
        return len(self._free_slots)

    @property
    def used_slots(self) -> int:
        return len(self._alloc_map)
