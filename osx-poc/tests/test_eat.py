"""Test M1 — Expert Access Table.

Tutti i test falliscono deterministicamente con NotImplementedError
fino all'implementazione in Sprint 1. Questo è il comportamento atteso.

Coverage target: > 95% (misurata con pytest-cov).
"""
import pytest
import threading
from eat import ExpertAccessTable, EATEntry, Tier
from eat.bloom import BloomFilter
from eat.slab import SlabAllocator


# ── BloomFilter ────────────────────────────────────────────────────────────────

class TestBloomFilter:

    def test_add_and_may_contain_expert(self):
        bf = BloomFilter(capacity=1000)
        with pytest.raises(NotImplementedError):
            bf.add(expert_id=0, shard_idx=0)

    def test_may_contain_shard_miss(self):
        bf = BloomFilter(capacity=1000)
        with pytest.raises(NotImplementedError):
            bf.may_contain_shard(expert_id=999, shard_idx=0)

    def test_false_positive_rate_within_spec(self):
        """FP rate < 1% su 10.000 lookup negativi — da implementare Sprint 1."""
        pytest.skip("TODO Sprint 1 — richiede implementazione BloomFilter")


# ── SlabAllocator ──────────────────────────────────────────────────────────────

class TestSlabAllocator:

    def test_initialize(self):
        slab = SlabAllocator(n_slots=2)
        with pytest.raises(NotImplementedError):
            slab.initialize()

    def test_alloc_free_cycle(self):
        pytest.skip("TODO Sprint 1 — richiede initialize()")

    def test_pool_exhaustion_raises(self):
        pytest.skip("TODO Sprint 1 — alloc oltre n_slots deve raise MemoryError")

    def test_variable_tail_shard(self):
        pytest.skip("TODO Sprint 1 — shard tail con size < SHARD_SIZE_BYTES")


# ── ExpertAccessTable ──────────────────────────────────────────────────────────

class TestEAT:

    @pytest.fixture
    def eat(self):
        return ExpertAccessTable(capacity=1000, n_slots=4)

    def test_insert_and_lookup(self, eat):
        with pytest.raises(NotImplementedError):
            eat.insert(expert_id=0, shard_idx=0, tier=Tier.NVME)

    def test_lookup_miss_returns_none(self, eat):
        with pytest.raises(NotImplementedError):
            eat.lookup(expert_id=999, shard_idx=0)

    def test_update_tier(self, eat):
        with pytest.raises(NotImplementedError):
            eat.update_tier(expert_id=0, shard_idx=0, new_tier=Tier.VRAM)

    def test_evict(self, eat):
        with pytest.raises(NotImplementedError):
            eat.evict(expert_id=0, shard_idx=0)

    def test_access_increments_count(self, eat):
        pytest.skip("TODO Sprint 1 — richiede insert() implementato")

    def test_get_tier(self, eat):
        with pytest.raises(NotImplementedError):
            eat.get_tier(Tier.VRAM)

    def test_len_empty(self, eat):
        assert len(eat) == 0   # questo funziona già — dict vuoto


# ── Thread safety ──────────────────────────────────────────────────────────────

class TestEATConcurrency:

    def test_concurrent_insert_no_race(self):
        """8 thread, 10.000 insert concorrenti — zero race condition."""
        pytest.skip("TODO Sprint 1 — richiede insert() implementato")

    def test_concurrent_read_write(self):
        """Read path lock-free durante write concorrenti."""
        pytest.skip("TODO Sprint 1")

    def test_version_counter_monotonic(self):
        """Version counter deve essere strettamente monotono con CAS."""
        pytest.skip("TODO Sprint 1")


# ── Benchmark smoke (non pytest-bench, solo ordine di grandezza) ───────────────

class TestEATLatencySmoke:

    def test_lookup_latency_order_of_magnitude(self):
        """Verifica che lookup() su dict Python sia < 10 µs — sanity check."""
        pytest.skip("TODO Sprint 1 — baseline latenza prima dell'ottimizzazione")
