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

    def test_rebuild_drops_entries_not_in_pairs(self):
        """rebuild() ricostruisce da zero — entry non ripassate spariscono dal BF."""
        bf = BloomFilter(capacity=1000)
        bf.add(expert_id=0, shard_idx=0)
        bf.add(expert_id=1, shard_idx=0)
        assert bf.may_contain_shard(0, 0) is True
        assert bf.may_contain_shard(1, 0) is True

        bf.rebuild(pairs=[(1, 0)])  # solo l'expert 1 sopravvive

        assert bf.may_contain_shard(0, 0) is False
        assert bf.may_contain_shard(1, 0) is True

    def test_rebuild_empty_pairs_clears_filter(self):
        bf = BloomFilter(capacity=1000)
        bf.add(expert_id=0, shard_idx=0)

        bf.rebuild(pairs=[])

        assert bf.may_contain_shard(0, 0) is False
        assert len(bf) == 0


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

    def test_evict_leaves_bloom_false_positive_until_rebuild(self):
        """Prima del rebuild periodico l'entry evicted resta un falso positivo nel
        Bloom filter (comportamento noto e accettato, GitHub issue #4) — dopo il
        rebuild sparisce."""
        eat = ExpertAccessTable(capacity=1000, n_slots=4, bloom_rebuild_every=3)
        eat.insert(expert_id=0, shard_idx=0)
        eat.evict(expert_id=0, shard_idx=0)

        # sotto la soglia di rebuild: il BF non è ancora stato ricostruito
        assert eat._bloom.may_contain_shard(0, 0) is True
        assert eat.stats()["bloom_evictions_since_rebuild"] == 1

    def test_evict_triggers_periodic_bloom_rebuild(self):
        eat = ExpertAccessTable(capacity=1000, n_slots=4, bloom_rebuild_every=3)
        for shard_idx in range(3):
            eat.insert(expert_id=0, shard_idx=shard_idx)
        for shard_idx in range(3):
            eat.evict(expert_id=0, shard_idx=shard_idx)  # 3a eviction fa scattare il rebuild

        assert eat.stats()["bloom_evictions_since_rebuild"] == 0
        for shard_idx in range(3):
            assert eat._bloom.may_contain_shard(0, shard_idx) is False

    def test_evict_rebuild_keeps_still_present_shards(self):
        """Il rebuild ricostruisce dalle chiavi correnti — gli shard non evicted
        restano trovabili via Bloom filter dopo il rebuild."""
        eat = ExpertAccessTable(capacity=1000, n_slots=4, bloom_rebuild_every=1)
        eat.insert(expert_id=0, shard_idx=0)
        eat.insert(expert_id=1, shard_idx=0)

        eat.evict(expert_id=0, shard_idx=0)  # rebuild_every=1: rebuild immediato

        assert eat._bloom.may_contain_shard(0, 0) is False
        assert eat._bloom.may_contain_shard(1, 0) is True
        assert eat.lookup(expert_id=1, shard_idx=0) is not None

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
