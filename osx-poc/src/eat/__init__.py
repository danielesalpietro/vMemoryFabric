"""M1 — Expert Access Table (EAT)

Slab allocator + Version manager. Bloom filter removed 2026-08-12
(issue #1) — measured consistently slower than a direct dict lookup at
this scale, see eat.py's module docstring.
Storage: DDR4 (numpy) — PMEM deferred.
Thread safety: RW lock + atomic CAS per version counter.
"""
from .types import ExpertID, ShardID, Tier, EATEntry
from .slab import SlabAllocator
from .eat import ExpertAccessTable

__all__ = ["ExpertAccessTable", "EATEntry", "ExpertID", "ShardID", "Tier",
           "SlabAllocator"]
