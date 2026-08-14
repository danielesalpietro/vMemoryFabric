"""Tipi fondamentali per M1 — EAT."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum

ExpertID = int   # 0..N_experts-1  (2 bytes in layout finale)
ShardID  = int   # 0..N_shards-1   (1 byte in layout finale)

SHARD_SIZE_MB: int = 256
SHARD_SIZE_BYTES: int = SHARD_SIZE_MB * 1024 * 1024


class Tier(IntEnum):
    """EMH tier identifiers — adattati al setup dev corrente (no PMEM).

    Layout fisico su Z8 G4 (dev target):
        EMH-1a  RTX 3090 VRAM    24 GB  (hot)
        EMH-1c  DDR4             256 GB (warm buffer)
        EMH-3   NVMe / volume    1 TB   (cold)

    PMEM (EMH-2) deferred — sarà inserito tra DDR4 e NVMe quando disponibile.
    """
    VRAM   = 0   # EMH-1a: RTX 3090
    DDR4   = 1   # EMH-1c: host RAM
    NVME   = 2   # EMH-3 : cold storage
    UNKNOWN = 99


@dataclass
class EATEntry:
    """Una riga della Expert Access Table.

    Layout target: 28 bytes/entry (vedi spec OSX v1.0 §2.2).
    In Python usiamo dataclass per chiarezza; il layout compatto
    verrà implementato con ctypes/struct nella fase di ottimizzazione.
    """
    expert_id:      ExpertID
    shard_idx:      ShardID
    tier:           Tier            = Tier.UNKNOWN
    access_count:   int             = 0             # 4 bytes
    last_access_ts: float           = field(default_factory=time.monotonic)  # 8 bytes
    semantic_vec_ptr: int | None = None          # 8 bytes (futuro: indice vettore)
    version:        int             = 0             # 4 bytes (CAS counter)

    def touch(self) -> None:
        """Aggiorna timestamp e contatore di accesso (non thread-safe — usa EAT.access())."""
        self.access_count += 1
        self.last_access_ts = time.monotonic()
