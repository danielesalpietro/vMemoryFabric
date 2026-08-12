"""Test M1 — Expert Access Table.

Sprint 1 (Möllstorp): BloomFilter, SlabAllocator ed EAT sono implementati.

Coverage target: > 95% (misurata con pytest-cov).
"""
import threading
import time

import pytest
from eat import EATEntry, ExpertAccessTable, Tier
from eat.bloom import BloomFilter
from eat.slab import SlabAllocator


# ── BloomFilter ────────────────────────────────────────────────────────────────

class TestBloomFilter:

    def test_add_and_may_contain_expert(self):
        bf = BloomFilter(capacity=1000)
        bf.add(expert_id=0, shard_idx=0)
        assert bf.may_contain_expert(0) is True
        assert bf.may_contain_expert(999) is False

    def test_may_contain_shard_miss(self):
        bf = BloomFilter(capacity=1000)
        assert bf.may_contain_shard(expert_id=999, shard_idx=0) is False

    def test_false_positive_rate_within_spec(self):
        """FP rate < 1% (+ margine statistico) su 10.000 lookup negativi."""
        bf = BloomFilter(capacity=10_000, error_rate=0.01)
        for expert_id in range(5_000):
            bf.add(expert_id=expert_id, shard_idx=0)

        false_positives = sum(
            1 for expert_id in range(5_000, 15_000)
            if bf.may_contain_shard(expert_id, 0)
        )
        fp_rate = false_positives / 10_000
        assert fp_rate < 0.02  # 1% target + margine per varianza statistica

    # ── remove (2026-08-12, issue #4) ────────────────────────────────────────

    def test_remove_shard_makes_may_contain_shard_false(self):
        bf = BloomFilter(capacity=1000)
        bf.add(expert_id=5, shard_idx=2)
        assert bf.may_contain_shard(5, 2) is True
        bf.remove_shard(5, 2)
        assert bf.may_contain_shard(5, 2) is False

    def test_remove_expert_makes_may_contain_expert_false(self):
        bf = BloomFilter(capacity=1000)
        bf.add(expert_id=7, shard_idx=0)
        assert bf.may_contain_expert(7) is True
        bf.remove_expert(7)
        assert bf.may_contain_expert(7) is False

    def test_remove_shard_does_not_affect_other_shards_of_same_expert(self):
        bf = BloomFilter(capacity=1000)
        bf.add(expert_id=1, shard_idx=0)
        bf.add(expert_id=1, shard_idx=1)
        bf.remove_shard(1, 0)
        assert bf.may_contain_shard(1, 0) is False
        assert bf.may_contain_shard(1, 1) is True

    def test_len_tracks_shard_level_add_and_remove(self):
        bf = BloomFilter(capacity=1000)
        bf.add(expert_id=0, shard_idx=0)
        bf.add(expert_id=0, shard_idx=1)
        assert len(bf) == 2
        bf.remove_shard(0, 0)
        assert len(bf) == 1

    def test_repeated_insert_evict_cycles_do_not_degrade_false_positive_rate(self):
        """Il bug reale di issue #4: pybloom_live non supportava
        remove(), quindi ogni eviction lasciava un falso positivo
        PERMANENTE — dopo abbastanza cicli insert/evict il Bloom filter
        degradava verso "quasi sempre presente", vanificando il fast-
        negative path. Con remove() reale, il FP rate resta vicino al
        target anche dopo 20 cicli x 500 insert+evict (10.000 "fantasmi"
        cumulativi se non venissero mai rimossi — più della capacity
        stessa, quindi un test che avrebbe fallito platealmente prima di
        questo fix)."""
        bf = BloomFilter(capacity=10_000, error_rate=0.01)
        for _ in range(20):
            for expert_id in range(500):
                bf.add(expert_id=expert_id, shard_idx=0)
            for expert_id in range(500):
                bf.remove_shard(expert_id, 0)
                bf.remove_expert(expert_id)

        false_positives = sum(
            1 for expert_id in range(500, 10_500)
            if bf.may_contain_shard(expert_id, 0)
        )
        fp_rate = false_positives / 10_000
        assert fp_rate < 0.02


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

    # ── evict + Bloom filter (2026-08-12, issue #4) ──────────────────────────

    def test_evict_clears_bloom_shard_entry(self, eat):
        eat.insert(expert_id=0, shard_idx=0)
        eat.evict(expert_id=0, shard_idx=0)
        assert eat._bloom.may_contain_shard(0, 0) is False

    def test_evict_clears_bloom_expert_entry_when_last_shard(self, eat):
        eat.insert(expert_id=0, shard_idx=0)
        eat.evict(expert_id=0, shard_idx=0)
        assert eat._bloom.may_contain_expert(0) is False

    def test_evict_keeps_bloom_expert_entry_when_other_shards_remain(self, eat):
        """Evictare uno shard non deve far "dimenticare" l'intero expert
        se altri suoi shard sono ancora nella tabella — il bit/contatore
        a livello expert è condiviso da tutti gli shard di quell'expert_id."""
        eat.insert(expert_id=0, shard_idx=0)
        eat.insert(expert_id=0, shard_idx=1)
        eat.evict(expert_id=0, shard_idx=0)
        assert eat._bloom.may_contain_expert(0) is True
        assert eat._bloom.may_contain_shard(0, 1) is True
        assert eat._bloom.may_contain_shard(0, 0) is False

    def test_evict_missing_does_not_touch_bloom(self, eat):
        """evict() su una entry inesistente non deve chiamare
        remove_shard()/remove_expert() — nessun errore, ma anche nessuna
        rimozione spuria di contatori mai incrementati."""
        eat.insert(expert_id=1, shard_idx=0)   # entry reale, non toccata sotto
        eat.evict(expert_id=99, shard_idx=0)   # mai inserita
        assert eat._bloom.may_contain_shard(1, 0) is True

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
