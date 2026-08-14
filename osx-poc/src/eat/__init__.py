"""M1 — Expert Access Table (EAT)

Bloom filter 2-livelli + Slab allocator + Version manager.
Storage: DDR4 (numpy) — PMEM deferred.
Thread safety: RW lock + atomic CAS per version counter.
"""
from .bloom import BloomFilter
from .eat import ExpertAccessTable
from .slab import SlabAllocator
from .types import EATEntry, ExpertID, ShardID, Tier

__all__ = ["ExpertAccessTable", "EATEntry", "ExpertID", "ShardID", "Tier",
           "BloomFilter", "SlabAllocator"]
