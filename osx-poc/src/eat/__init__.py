"""M1 — Expert Access Table (EAT)

Bloom filter 2-livelli + Slab allocator + Version manager.
Storage: DDR4 (numpy) — PMEM deferred.
Thread safety: RW lock + atomic CAS per version counter.
"""
from .types import ExpertID, ShardID, Tier, EATEntry
from .bloom import BloomFilter
from .slab import SlabAllocator
from .eat import ExpertAccessTable

__all__ = ["ExpertAccessTable", "EATEntry", "ExpertID", "ShardID", "Tier",
           "BloomFilter", "SlabAllocator"]
