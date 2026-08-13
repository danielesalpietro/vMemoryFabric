"""Test M1 — Expert Access Table.

Sprint 1 (Möllstorp): SlabAllocator ed EAT sono implementati. Bloom
filter (era qui, TestBloomFilter) rimosso 2026-08-12 insieme a
src/eat/bloom.py — issue #1, misurato consistentemente più lento di un
lookup diretto sul dict a questa scala, vedi LOGBOOK.md.

Coverage target: > 95% (misurata con pytest-cov).
"""
import threading
import time

import pytest
from eat import EATEntry, ExpertAccessTable, Tier
from eat.slab import SlabAllocator


# ── SlabAllocator ──────────────────────────────────────────────────────────────

class TestSlabAllocator:

    def test_initialize(self):
        slab = SlabAllocator(n_slots=2)
        assert slab.free_slots == 2
        assert slab.used_slots == 0
        slab.initialize()
        assert slab.free_slots == 2
        assert slab.used_slots == 0

    def test_alloc_free_cycle(self):
        slab = SlabAllocator(n_slots=2)
        slab.initialize()

        slot_idx = slab.alloc(expert_id=0, shard_idx=0, size_bytes=slab._shard_size)
        assert slab.used_slots == 1
        assert slab.free_slots == 1

        buf = slab.get_buffer(slot_idx)
        assert buf.shape == (slab._shard_size,)

        slab.free(slot_idx)
        assert slab.used_slots == 0
        assert slab.free_slots == 2

    def test_pool_exhaustion_raises(self):
        slab = SlabAllocator(n_slots=2, shard_size=1024)
        slab.initialize()

        slab.alloc(expert_id=0, shard_idx=0, size_bytes=1024)
        slab.alloc(expert_id=1, shard_idx=0, size_bytes=1024)

        with pytest.raises(MemoryError):
            slab.alloc(expert_id=2, shard_idx=0, size_bytes=1024)

    def test_variable_tail_shard(self):
        slab = SlabAllocator(n_slots=2, shard_size=1024)
        slab.initialize()

        slot_idx = slab.alloc(expert_id=0, shard_idx=1, size_bytes=100, is_tail=True)
        meta = slab._alloc_map[slot_idx]
        assert meta.size_bytes == 100
        assert meta.is_tail is True
        # slot fisico resta a dimensione piena — solo il payload logico è variabile
        assert slab.get_buffer(slot_idx).shape == (1024,)


# ── ExpertAccessTable ──────────────────────────────────────────────────────────

class TestEAT:

    @pytest.fixture
    def eat(self):
        return ExpertAccessTable(capacity=1000, n_slots=4)

    def test_insert_and_lookup(self, eat):
        entry = eat.insert(expert_id=0, shard_idx=0, tier=Tier.NVME)
        assert isinstance(entry, EATEntry)
        assert entry.expert_id == 0
        assert entry.tier == Tier.NVME

        looked_up = eat.lookup(expert_id=0, shard_idx=0)
        assert looked_up is entry
        assert len(eat) == 1

    def test_insert_duplicate_raises(self, eat):
        eat.insert(expert_id=0, shard_idx=0)
        with pytest.raises(KeyError):
            eat.insert(expert_id=0, shard_idx=0)

    def test_lookup_miss_returns_none(self, eat):
        assert eat.lookup(expert_id=999, shard_idx=0) is None

    def test_update_tier(self, eat):
        eat.insert(expert_id=0, shard_idx=0, tier=Tier.NVME)
        eat.update_tier(expert_id=0, shard_idx=0, new_tier=Tier.VRAM)

        entry = eat.lookup(expert_id=0, shard_idx=0)
        assert entry.tier == Tier.VRAM
        assert entry.version == 1

    def test_update_tier_missing_raises(self, eat):
        with pytest.raises(KeyError):
            eat.update_tier(expert_id=0, shard_idx=0, new_tier=Tier.VRAM)

    def test_evict(self, eat):
        eat.insert(expert_id=0, shard_idx=0)
        evicted = eat.evict(expert_id=0, shard_idx=0)

        assert evicted.expert_id == 0
        assert eat.lookup(expert_id=0, shard_idx=0) is None
        assert len(eat) == 0

    def test_evict_missing_returns_none(self, eat):
        assert eat.evict(expert_id=0, shard_idx=0) is None

    def test_access_increments_count(self, eat):
        eat.insert(expert_id=0, shard_idx=0)
        eat.access(expert_id=0, shard_idx=0)
        entry = eat.access(expert_id=0, shard_idx=0)

        assert entry.access_count == 2

    def test_access_missing_returns_none(self, eat):
        assert eat.access(expert_id=0, shard_idx=0) is None

    def test_get_tier(self, eat):
        eat.insert(expert_id=0, shard_idx=0, tier=Tier.VRAM)
        eat.insert(expert_id=1, shard_idx=0, tier=Tier.NVME)
        eat.insert(expert_id=2, shard_idx=0, tier=Tier.VRAM)

        vram_entries = eat.get_tier(Tier.VRAM)
        assert {e.expert_id for e in vram_entries} == {0, 2}

    def test_eviction_candidates_lru_order(self, eat):
        eat.insert(expert_id=0, shard_idx=0, tier=Tier.DDR4)
        eat.insert(expert_id=1, shard_idx=0, tier=Tier.DDR4)
        eat.insert(expert_id=2, shard_idx=0, tier=Tier.DDR4)

        eat.access(expert_id=1, shard_idx=0)  # ringiovanisce expert 1

        candidates = eat.eviction_candidates(Tier.DDR4, n=2)
        assert [c.expert_id for c in candidates] == [0, 2]

    def test_hottest_candidates_by_access_count(self, eat):
        eat.insert(expert_id=0, shard_idx=0, tier=Tier.DDR4)
        eat.insert(expert_id=1, shard_idx=0, tier=Tier.DDR4)
        eat.insert(expert_id=2, shard_idx=0, tier=Tier.DDR4)

        eat.access(expert_id=1, shard_idx=0)
        eat.access(expert_id=1, shard_idx=0)
        eat.access(expert_id=2, shard_idx=0)

        candidates = eat.hottest_candidates(Tier.DDR4, n=2)
        assert [c.expert_id for c in candidates] == [1, 2]

    def test_hottest_candidates_tie_break_by_recency(self, eat):
        eat.insert(expert_id=0, shard_idx=0, tier=Tier.DDR4)
        eat.insert(expert_id=1, shard_idx=0, tier=Tier.DDR4)

        eat.access(expert_id=0, shard_idx=0)   # stesso access_count (1) per entrambi...
        eat.access(expert_id=1, shard_idx=0)   # ...ma expert 1 acceduto più di recente

        candidates = eat.hottest_candidates(Tier.DDR4, n=2)
        assert [c.expert_id for c in candidates] == [1, 0]

    def test_hottest_candidates_ignores_other_tiers(self, eat):
        eat.insert(expert_id=0, shard_idx=0, tier=Tier.VRAM)
        eat.insert(expert_id=1, shard_idx=0, tier=Tier.DDR4)
        eat.access(expert_id=0, shard_idx=0)

        candidates = eat.hottest_candidates(Tier.DDR4, n=5)
        assert [c.expert_id for c in candidates] == [1]

    def test_hottest_candidates_empty_tier(self, eat):
        assert eat.hottest_candidates(Tier.VRAM, n=5) == []

    def test_len_empty(self, eat):
        assert len(eat) == 0   # dict vuoto

    def test_initialize_shutdown(self, eat):
        eat.initialize()
        assert eat._slab.free_slots == 4

        eat.insert(expert_id=0, shard_idx=0)
        eat.shutdown()

        assert len(eat) == 0
        assert eat._slab.free_slots == 4


# ── Thread safety ──────────────────────────────────────────────────────────────

class TestEATConcurrency:

    def test_concurrent_insert_no_race(self):
        """8 thread, 10.000 insert concorrenti — zero race condition."""
        eat = ExpertAccessTable(capacity=20_000, n_slots=4)
        n_threads = 8
        inserts_per_thread = 1_250  # 8 * 1250 = 10 000
        errors: list[Exception] = []
        lock = threading.Lock()

        def worker(thread_idx: int) -> None:
            try:
                for i in range(inserts_per_thread):
                    shard_idx = thread_idx * inserts_per_thread + i
                    eat.insert(expert_id=0, shard_idx=shard_idx)
            except Exception as exc:  # pragma: no cover - failure path
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(eat) == n_threads * inserts_per_thread

    def test_concurrent_read_write(self):
        """Read path stabile durante write concorrenti — nessuna eccezione/deadlock."""
        eat = ExpertAccessTable(capacity=20_000, n_slots=4)
        errors: list[Exception] = []
        lock = threading.Lock()
        stop = threading.Event()

        def writer() -> None:
            try:
                for shard_idx in range(2_000):
                    eat.insert(expert_id=0, shard_idx=shard_idx)
            except Exception as exc:  # pragma: no cover - failure path
                with lock:
                    errors.append(exc)
            finally:
                stop.set()

        def reader() -> None:
            try:
                while not stop.is_set():
                    eat.lookup(expert_id=0, shard_idx=0)
                    eat.get_tier(Tier.NVME)
            except Exception as exc:  # pragma: no cover - failure path
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == []
        assert len(eat) == 2_000

    def test_version_counter_monotonic(self):
        """Version counter deve essere strettamente monotono con CAS."""
        eat = ExpertAccessTable(capacity=1000, n_slots=4)
        eat.insert(expert_id=0, shard_idx=0, tier=Tier.NVME)

        n_threads = 8
        updates_per_thread = 200
        tiers = [Tier.NVME, Tier.DDR4, Tier.VRAM]

        def worker(thread_idx: int) -> None:
            for i in range(updates_per_thread):
                eat.update_tier(expert_id=0, shard_idx=0, new_tier=tiers[i % len(tiers)])

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        entry = eat.lookup(expert_id=0, shard_idx=0)
        assert entry.version == n_threads * updates_per_thread


# ── Locking strategies (issue #23: A=single, B=striped, C=lockfree_read) ───────

class TestEATLockingStrategies:

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            ExpertAccessTable(capacity=100, n_slots=1, locking_strategy="bogus")

    @pytest.mark.parametrize("strategy", ["single", "striped", "lockfree_read"])
    def test_crud_roundtrip(self, strategy):
        eat = ExpertAccessTable(capacity=1000, n_slots=4, locking_strategy=strategy, n_shards=4)

        entry = eat.insert(expert_id=0, shard_idx=0, tier=Tier.NVME)
        assert eat.lookup(expert_id=0, shard_idx=0) is entry

        eat.update_tier(expert_id=0, shard_idx=0, new_tier=Tier.VRAM)
        assert eat.lookup(expert_id=0, shard_idx=0).tier == Tier.VRAM

        touched = eat.access(expert_id=0, shard_idx=0)
        assert touched.access_count == 1

        assert eat.evict(expert_id=0, shard_idx=0) is not None
        assert eat.lookup(expert_id=0, shard_idx=0) is None
        assert len(eat) == 0

    @pytest.mark.parametrize("strategy", ["single", "striped", "lockfree_read"])
    def test_bulk_ops_span_shards(self, strategy):
        """Entry su expert_id diversi finiscono su shard diversi con
        locking_strategy="striped" — i metodi bulk devono comunque vederle
        tutte (snapshot rilassato ma completo, non parziale)."""
        eat = ExpertAccessTable(capacity=1000, n_slots=4, locking_strategy=strategy, n_shards=4)
        for expert_id in range(8):
            eat.insert(expert_id=expert_id, shard_idx=0, tier=Tier.DDR4)
        eat.access(expert_id=3, shard_idx=0)
        eat.access(expert_id=3, shard_idx=0)

        assert len(eat.get_tier(Tier.DDR4)) == 8
        assert len(eat) == 8
        assert eat.stats()["total_entries"] == 8
        assert eat.hottest_candidates(Tier.DDR4, n=1)[0].expert_id == 3
        assert len(eat.eviction_candidates(Tier.DDR4, n=8)) == 8

    def test_concurrent_insert_no_race_striped(self):
        """Stesso scenario di TestEATConcurrency.test_concurrent_insert_no_race,
        ma con locking_strategy="striped" — deve restare senza race condition."""
        eat = ExpertAccessTable(capacity=20_000, n_slots=4, locking_strategy="striped", n_shards=8)
        n_threads = 8
        inserts_per_thread = 1_250
        errors: list[Exception] = []
        lock = threading.Lock()

        def worker(thread_idx: int) -> None:
            try:
                for i in range(inserts_per_thread):
                    shard_idx = thread_idx * inserts_per_thread + i
                    eat.insert(expert_id=0, shard_idx=shard_idx)
            except Exception as exc:  # pragma: no cover - failure path
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(eat) == n_threads * inserts_per_thread


# ── Benchmark smoke (non pytest-bench, solo ordine di grandezza) ───────────────

class TestEATLatencySmoke:

    def test_lookup_latency_order_of_magnitude(self):
        """Verifica che lookup() su dict Python sia dell'ordine di grandezza atteso."""
        eat = ExpertAccessTable(capacity=1000, n_slots=4)
        eat.insert(expert_id=0, shard_idx=0)

        n_iterations = 10_000
        # warm-up
        for _ in range(100):
            eat.lookup(expert_id=0, shard_idx=0)

        start = time.perf_counter()
        for _ in range(n_iterations):
            eat.lookup(expert_id=0, shard_idx=0)
        elapsed = time.perf_counter() - start

        avg_latency_s = elapsed / n_iterations
        # Baseline pre-ottimizzazione (dict Python + bloom check, no C layout):
        # target documentato è < 100ns per il bloom hit + dict access reale, ma
        # in CI (host condivisi, no isolamento) usiamo un margine più permissivo.
        assert avg_latency_s < 50e-6  # 50 µs — sanity check, non un benchmark
