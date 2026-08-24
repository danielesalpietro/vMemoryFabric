"""Tipi fondamentali per M1 — EAT."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum

ExpertID = int   # 0..N_experts-1  (2 bytes in layout finale)
ShardID  = int   # 0..N_shards-1   (1 byte in layout finale)

SHARD_SIZE_MB: int = 256
SHARD_SIZE_BYTES: int = SHARD_SIZE_MB * 1024 * 1024


class Tier(IntEnum):
    """EMH tier identifiers.

    Layout fisico su Z8 G4 bare-metal (osx-poc/LOGBOOK_NEW_Z8.md, "passo 4"):
        EMH-1a  RTX 3090 VRAM    24 GB  (hot)
        EMH-1c  DDR4             ~236 GB (warm buffer)
        EMH-2   Optane PMEM      252 GB (fsdax, tra DDR4 e NVMe)
        EMH-3   NVMe / volume    1 TB   (cold)

    PMEM = 3, non inserito numericamente tra DDR4 (1) e NVME (2), per non
    rinumerare NVME su un enum già in uso altrove (log/telemetria) — la
    posizione "tra DDR4 e NVMe" è nel percorso di promote()/evict()
    (tier/manager.py), non nel valore numerico dell'IntEnum.
    """
    VRAM   = 0   # EMH-1a: RTX 3090
    DDR4   = 1   # EMH-1c: host RAM
    NVME   = 2   # EMH-3 : cold storage
    PMEM   = 3   # EMH-2 : Optane DC Persistent Memory (fsdax)
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
    version:        int             = 0             # 4 bytes (seqlock/CAS counter)

    @property
    def write_in_progress(self) -> bool:
        """True se in questo istante è in corso una scrittura su questa entry
        (version dispari — protocollo seqlock, issue #23).

        Pensato per EAT.lookup() con locking_strategy="lockfree_read": chi
        legge senza lock può controllare version prima e dopo aver letto i
        campi che gli interessano — se è dispari, o se è cambiata tra le
        due letture, il dato letto può essere incoerente (es. touch() a
        metà: access_count già aggiornato, last_access_ts non ancora) e
        chi legge decide da sé cosa fare (retry, scarta, accetta comunque,
        logga). Questo non è enforcement automatico: è il segnale grezzo
        su cui costruire la policy.
        """
        return self.version % 2 == 1

    @contextmanager
    def seqlock_write(self):
        """Contesto per mutare l'entry sotto protocollo seqlock: incrementa
        version prima (dispari = scrittura in corso) e dopo (pari =
        stabile) del blocco. Non sostituisce il lock dello shard di
        EAT — i writer restano serializzati da quello; questo protegge
        solo i lettori lock-free che leggono senza mai prendere un lock.
        """
        self.version += 1
        try:
            yield
        finally:
            self.version += 1

    def touch(self) -> None:
        """Aggiorna timestamp e contatore di accesso (non thread-safe — usa EAT.access())."""
        with self.seqlock_write():
            self.access_count += 1
            self.last_access_ts = time.monotonic()
