"""Test M2 — EMH Tier Manager.

Coverage target: > 90%.

Split mirrors the CI cpu-tests / full-gpu-tests jobs: anything that
touches real CUDA (GPUTransfer.to_vram/to_ddr4/vram_*, DDR4->VRAM and
NVME->VRAM promotion, evict_to_free_vram, VRAM-side integration) is
@pytest.mark.gpu and only runs for real on the self-hosted
Z8-G4-RTX3090 runner. Everything else (NVMe I/O, slab bookkeeping,
policies, NVME->DDR4 promotion, the concurrency regression test) runs
anywhere torch is importable.
"""
import asyncio
import logging
import time

import numpy as np
import pytest

from eat import ExpertAccessTable, Tier
from eat.types import EATEntry
from tier import AsyncNVMeIO, GPUTransfer, LRUPolicy, SEEPolicy, TierManager


def _write_shard_file(base_path, expert_id, shard_idx, payload=b"x" * 128):
    path = base_path / str(expert_id) / f"{shard_idx}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


# ── AsyncNVMeIO ────────────────────────────────────────────────────────────────

class TestAsyncNVMeIO:

    @pytest.mark.asyncio
    async def test_read_shard_missing_raises(self, tmp_path):
        io = AsyncNVMeIO(base_path=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            await io.read_shard(expert_id=0, shard_idx=0)

    @pytest.mark.asyncio
    async def test_write_then_read_roundtrip(self, tmp_path):
        io = AsyncNVMeIO(base_path=str(tmp_path))
        payload = np.arange(256, dtype=np.uint8)
        await io.write_shard(expert_id=0, shard_idx=0, data=payload)
        result = await io.read_shard(expert_id=0, shard_idx=0)
        np.testing.assert_array_equal(result, payload)

    @pytest.mark.asyncio
    async def test_read_shard_into_out_buffer(self, tmp_path):
        io = AsyncNVMeIO(base_path=str(tmp_path))
        payload = np.arange(64, dtype=np.uint8)
        await io.write_shard(expert_id=0, shard_idx=0, data=payload)
        out = np.zeros(64, dtype=np.uint8)
        result = await io.read_shard(expert_id=0, shard_idx=0, out=out)
        assert result is out
        np.testing.assert_array_equal(out, payload)

    @pytest.mark.asyncio
    async def test_exists_false_before_write_true_after(self, tmp_path):
        io = AsyncNVMeIO(base_path=str(tmp_path))
        assert await io.exists(expert_id=0, shard_idx=0) is False
        await io.write_shard(expert_id=0, shard_idx=0, data=np.zeros(4, dtype=np.uint8))
        assert await io.exists(expert_id=0, shard_idx=0) is True

    @pytest.mark.asyncio
    async def test_delete_removes_shard(self, tmp_path):
        io = AsyncNVMeIO(base_path=str(tmp_path))
        await io.write_shard(expert_id=0, shard_idx=0, data=np.zeros(4, dtype=np.uint8))
        await io.delete(expert_id=0, shard_idx=0)
        assert await io.exists(expert_id=0, shard_idx=0) is False


# ── GPUTransfer ────────────────────────────────────────────────────────────────

@pytest.mark.gpu
class TestGPUTransfer:

    @pytest.fixture
    def gpu(self):
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("CUDA non disponibile")
            return GPUTransfer(device_id=0)
        except RuntimeError as e:
            pytest.skip(f"GPU non disponibile: {e}")

    def test_vram_total_is_24gb(self, gpu):
        total_gb = gpu.vram_total_bytes() / (1024 ** 3)
        assert total_gb > 20  # RTX 3090 nominale 24GB, overhead driver escluso

    def test_to_vram_and_back(self, gpu):
        data = np.arange(256, dtype=np.uint8)
        tensor = gpu.to_vram(data)
        assert tensor.is_cuda
        back = gpu.to_ddr4(tensor)
        np.testing.assert_array_equal(back, data)

    def test_vram_free_decreases_after_load(self, gpu):
        free_before = gpu.vram_free_bytes()
        data = np.zeros(64 * 1024 * 1024, dtype=np.uint8)  # 64MB
        tensor = gpu.to_vram(data)
        free_after = gpu.vram_free_bytes()
        assert free_after < free_before
        del tensor


# ── SEE / LRU Policy ──────────────────────────────────────────────────────────

class TestEvictionPolicies:

    def _entries(self):
        now = time.monotonic()
        e_old = EATEntry(expert_id=0, shard_idx=0, tier=Tier.VRAM,
                          access_count=1, last_access_ts=now - 100)
        e_new = EATEntry(expert_id=0, shard_idx=1, tier=Tier.VRAM,
                          access_count=1, last_access_ts=now - 1)
        return e_old, e_new, now

    def test_lru_rank_by_last_access(self):
        lru = LRUPolicy()
        e_old, e_new, _ = self._entries()
        ranked = lru.rank(candidates=[e_new, e_old], n=2)
        assert [c.entry for c in ranked] == [e_old, e_new]  # meno recente prima

    def test_lru_rank_empty(self):
        lru = LRUPolicy()
        assert lru.rank(candidates=[], n=1) == []

    def test_see_score_without_context_fallback_lru(self):
        see = SEEPolicy()
        e_old, e_new, now = self._entries()
        score_old = see.score(e_old, context_vec=None, now=now)
        score_new = see.score(e_new, context_vec=None, now=now)
        assert score_new > score_old  # più recente -> score più alto -> si mantiene

    def test_see_rank_orders_oldest_first(self):
        see = SEEPolicy()
        e_old, e_new, _ = self._entries()
        ranked = see.rank(candidates=[e_new, e_old], n=2)
        assert [c.entry for c in ranked] == [e_old, e_new]

    def test_see_weights_sum_to_one(self):
        see = SEEPolicy(alpha=0.3, beta=0.3, gamma=0.4)
        assert abs(see.alpha + see.beta + see.gamma - 1.0) < 1e-6

    def test_see_invalid_weights_raises(self):
        with pytest.raises(AssertionError):
            SEEPolicy(alpha=0.5, beta=0.5, gamma=0.5)


# ── TierManager ────────────────────────────────────────────────────────────────

class TestTierManager:

    @pytest.fixture
    def manager(self, tmp_path):
        eat = ExpertAccessTable(capacity=100, n_slots=2)
        eat.initialize()
        mgr = TierManager(eat=eat, nvme_path=str(tmp_path), gpu_device=0)
        yield mgr
        eat.shutdown()

    @pytest.mark.asyncio
    async def test_promote_nvme_to_ddr4(self, manager, tmp_path):
        manager._eat.insert(expert_id=0, shard_idx=0, tier=Tier.NVME)
        _write_shard_file(tmp_path, 0, 0)
        latency = await manager.promote(expert_id=0, shard_idx=0, target_tier=Tier.DDR4)
        assert latency >= 0
        assert manager._eat.lookup(0, 0).tier == Tier.DDR4
        assert manager._eat.slab.used_slots == 1

    @pytest.mark.asyncio
    async def test_promote_missing_shard_raises(self, manager):
        with pytest.raises(ValueError):
            await manager.promote(expert_id=99, shard_idx=0, target_tier=Tier.DDR4)

    @pytest.mark.asyncio
    async def test_promote_same_tier_raises(self, manager, tmp_path):
        manager._eat.insert(expert_id=0, shard_idx=0, tier=Tier.NVME)
        _write_shard_file(tmp_path, 0, 0)
        with pytest.raises(ValueError):
            await manager.promote(expert_id=0, shard_idx=0, target_tier=Tier.NVME)

    @pytest.mark.asyncio
    async def test_evict_single_hop_only_raises_from_nvme(self, manager):
        """evict() non incatena hop multipli: da NVME non c'è tier inferiore
        raggiungibile, e a differenza di promote() questo NON viene esteso
        automaticamente — deve sollevare ValueError."""
        manager._eat.insert(expert_id=0, shard_idx=0, tier=Tier.NVME)
        with pytest.raises(ValueError):
            await manager.evict(expert_id=0, shard_idx=0)

    @pytest.mark.asyncio
    async def test_concurrent_double_promote_no_slab_leak(self, manager, tmp_path):
        """Regression: due promote() concorrenti sulla stessa key non devono
        allocare due slot slab per un solo shard logico (vedi per-key lock
        in TierManager._lock_for)."""
        manager._eat.insert(expert_id=0, shard_idx=0, tier=Tier.NVME)
        _write_shard_file(tmp_path, 0, 0)
        results = await asyncio.gather(
            manager.promote(expert_id=0, shard_idx=0, target_tier=Tier.DDR4),
            manager.promote(expert_id=0, shard_idx=0, target_tier=Tier.DDR4),
            return_exceptions=True,
        )
        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]
        assert len(successes) == 1
        assert len(failures) == 1
        assert isinstance(failures[0], ValueError)
        assert manager._eat.slab.used_slots == 1

    @pytest.mark.asyncio
    async def test_prefetch_logs_failures_without_raising(self, manager, caplog):
        with caplog.at_level(logging.WARNING):
            await manager.prefetch([(0, 0), (0, 1)])  # nessuno presente in EAT
        assert caplog.text.count("prefetch fallito") == 2

    def test_eat_property_exposes_underlying_instance(self, manager):
        """eat (2026-08-12, issue #17) espone la stessa istanza passata al
        costruttore, non una copia — stesso pattern/contratto di
        ExpertAccessTable.slab."""
        assert manager.eat is manager._eat


@pytest.mark.gpu
class TestTierManagerGPU:

    @pytest.fixture
    def manager(self, tmp_path):
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA non disponibile")
        eat = ExpertAccessTable(capacity=100, n_slots=2)
        eat.initialize()
        mgr = TierManager(eat=eat, nvme_path=str(tmp_path), gpu_device=0)
        yield mgr
        eat.shutdown()

    @pytest.mark.asyncio
    async def test_promote_ddr4_to_vram(self, manager, tmp_path):
        manager._eat.insert(expert_id=0, shard_idx=0, tier=Tier.NVME)
        _write_shard_file(tmp_path, 0, 0)
        await manager.promote(expert_id=0, shard_idx=0, target_tier=Tier.DDR4)
        latency = await manager.promote(expert_id=0, shard_idx=0, target_tier=Tier.VRAM)
        assert latency >= 0
        assert manager._eat.lookup(0, 0).tier == Tier.VRAM
        assert manager._eat.slab.used_slots == 0  # DDR4 era solo staging

    @pytest.mark.asyncio
    async def test_promote_nvme_to_vram_chained(self, manager, tmp_path):
        manager._eat.insert(expert_id=1, shard_idx=0, tier=Tier.NVME)
        _write_shard_file(tmp_path, 1, 0)
        await manager.promote(expert_id=1, shard_idx=0, target_tier=Tier.VRAM)
        assert manager._eat.lookup(1, 0).tier == Tier.VRAM

    @pytest.mark.asyncio
    async def test_evict_vram_to_ddr4(self, manager, tmp_path):
        manager._eat.insert(expert_id=2, shard_idx=0, tier=Tier.NVME)
        _write_shard_file(tmp_path, 2, 0)
        await manager.promote(expert_id=2, shard_idx=0, target_tier=Tier.VRAM)
        await manager.evict(expert_id=2, shard_idx=0)
        assert manager._eat.lookup(2, 0).tier == Tier.DDR4

    @pytest.mark.asyncio
    async def test_evict_frees_vram_visible_to_mem_get_info(self, manager, tmp_path):
        """Regressione: evict() da VRAM deve liberare memoria visibile a
        vram_free_bytes() (torch.cuda.mem_get_info), non solo alla EAT —
        senza GPUTransfer.empty_cache() il caching allocator di PyTorch la
        trattiene per riuso e vram_free_bytes() resta invariato, facendo
        ciclare all'infinito evict_to_free_vram(). Trovato eseguendo i test
        su hardware reale, mai riprodotto senza CUDA vera."""
        manager._eat.insert(expert_id=4, shard_idx=0, tier=Tier.NVME)
        _write_shard_file(tmp_path, 4, 0, payload=bytes(64 * 1024 * 1024))  # 64MB, misurabile
        await manager.promote(expert_id=4, shard_idx=0, target_tier=Tier.VRAM)
        free_with_shard = manager._gpu.vram_free_bytes()
        await manager.evict(expert_id=4, shard_idx=0)
        free_after_evict = manager._gpu.vram_free_bytes()
        assert free_after_evict > free_with_shard

    @pytest.mark.asyncio
    async def test_evict_to_free_vram_raises_when_tier_empty(self, manager):
        with pytest.raises(MemoryError):
            await manager.evict_to_free_vram(
                target_free_bytes=manager._gpu.vram_total_bytes() * 2
            )

    @pytest.mark.asyncio
    async def test_evict_to_free_vram_evicts_candidate(self, manager, tmp_path):
        manager._eat.insert(expert_id=3, shard_idx=0, tier=Tier.NVME)
        _write_shard_file(tmp_path, 3, 0)
        await manager.promote(expert_id=3, shard_idx=0, target_tier=Tier.VRAM)
        target = manager._gpu.vram_free_bytes() + 1
        evicted = await manager.evict_to_free_vram(target_free_bytes=target)
        assert len(evicted) >= 1
        assert manager._eat.lookup(3, 0).tier == Tier.DDR4

    # ── promote_live_tensor (2026-08-12, issue #17) ──────────────────────────

    @pytest.mark.asyncio
    async def test_promote_live_tensor_seeds_eat_and_promotes(self, manager):
        data = np.arange(256, dtype=np.uint8)
        tensor = await manager.promote_live_tensor(expert_id=5, shard_idx=0, cpu_data=data)
        assert tensor.is_cuda
        entry = manager._eat.lookup(5, 0)
        assert entry is not None
        assert entry.tier == Tier.VRAM
        np.testing.assert_array_equal(manager._gpu.to_ddr4(tensor), data)

    @pytest.mark.asyncio
    async def test_promote_live_tensor_idempotent(self, manager):
        data = np.arange(128, dtype=np.uint8)
        first = await manager.promote_live_tensor(expert_id=6, shard_idx=0, cpu_data=data)
        second = await manager.promote_live_tensor(expert_id=6, shard_idx=0, cpu_data=data)
        assert first is second   # no ri-transfer, stesso tensore ritornato

    @pytest.mark.asyncio
    async def test_promote_live_tensor_accepts_cpu_torch_tensor(self, manager):
        import torch
        data = torch.arange(64, dtype=torch.uint8)
        tensor = await manager.promote_live_tensor(expert_id=7, shard_idx=0, cpu_data=data)
        assert tensor.is_cuda
        assert tensor.shape == data.shape

    @pytest.mark.asyncio
    async def test_promote_live_tensor_wrong_existing_tier_raises(self, manager):
        manager._eat.insert(expert_id=8, shard_idx=0, tier=Tier.NVME)
        with pytest.raises(ValueError):
            await manager.promote_live_tensor(
                expert_id=8, shard_idx=0, cpu_data=np.zeros(8, dtype=np.uint8),
            )

    @pytest.mark.asyncio
    async def test_promote_live_tensor_with_pin_true(self, manager):
        """pin=True — verificato sicuro dal soak test 2026-08-12 (LOGBOOK.md),
        qui solo che il flag non rompa il transfer e produca lo stesso
        risultato numerico di pin=False."""
        data = np.arange(256, dtype=np.uint8)
        tensor = await manager.promote_live_tensor(
            expert_id=9, shard_idx=0, cpu_data=data, pin=True,
        )
        assert tensor.is_cuda
        np.testing.assert_array_equal(manager._gpu.to_ddr4(tensor), data)


# ── Integration: EAT ↔ Tier Manager ──────────────────────────────────────────

class TestEATTierIntegration:

    @pytest.mark.asyncio
    async def test_promote_updates_eat_tier(self, tmp_path):
        """Post-promozione, EAT.lookup deve restituire il nuovo tier."""
        eat = ExpertAccessTable(capacity=10, n_slots=2)
        eat.initialize()
        mgr = TierManager(eat=eat, nvme_path=str(tmp_path), gpu_device=0)
        eat.insert(expert_id=0, shard_idx=0, tier=Tier.NVME)
        _write_shard_file(tmp_path, 0, 0)
        await mgr.promote(expert_id=0, shard_idx=0, target_tier=Tier.DDR4)
        assert eat.lookup(0, 0).tier == Tier.DDR4
        eat.shutdown()

    @pytest.mark.gpu
    @pytest.mark.asyncio
    async def test_evict_updates_eat_tier(self, tmp_path):
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA non disponibile")
        eat = ExpertAccessTable(capacity=10, n_slots=2)
        eat.initialize()
        mgr = TierManager(eat=eat, nvme_path=str(tmp_path), gpu_device=0)
        eat.insert(expert_id=0, shard_idx=0, tier=Tier.NVME)
        _write_shard_file(tmp_path, 0, 0)
        await mgr.promote(expert_id=0, shard_idx=0, target_tier=Tier.VRAM)
        await mgr.evict(expert_id=0, shard_idx=0)
        assert eat.lookup(0, 0).tier == Tier.DDR4
        eat.shutdown()
