"""Test M2 — EMH Tier Manager.

Coverage target: > 90%.
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from eat import ExpertAccessTable, Tier
from tier import TierManager, SEEPolicy, LRUPolicy, AsyncNVMeIO, GPUTransfer


# ── AsyncNVMeIO ────────────────────────────────────────────────────────────────

class TestAsyncNVMeIO:

    @pytest.mark.asyncio
    async def test_read_shard_not_exists(self, tmp_path):
        io = AsyncNVMeIO(base_path=str(tmp_path))
        with pytest.raises(NotImplementedError):
            await io.read_shard(expert_id=0, shard_idx=0)

    @pytest.mark.asyncio
    async def test_write_then_read_roundtrip(self, tmp_path):
        pytest.skip("TODO Sprint 2 — richiede read/write implementati")

    @pytest.mark.asyncio
    async def test_exists_false_before_write(self, tmp_path):
        io = AsyncNVMeIO(base_path=str(tmp_path))
        with pytest.raises(NotImplementedError):
            await io.exists(expert_id=0, shard_idx=0)


# ── GPUTransfer ────────────────────────────────────────────────────────────────

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
        with pytest.raises(NotImplementedError):
            gpu.vram_total_bytes()

    def test_to_vram_and_back(self, gpu):
        pytest.skip("TODO Sprint 2 — richiede to_vram() implementato")

    def test_vram_free_decreases_after_load(self, gpu):
        pytest.skip("TODO Sprint 2")


# ── SEE / LRU Policy ──────────────────────────────────────────────────────────

class TestEvictionPolicies:

    def test_lru_rank_by_last_access(self):
        lru = LRUPolicy()
        with pytest.raises(NotImplementedError):
            lru.rank(candidates=[], n=1)

    def test_see_score_without_context_fallback_lru(self):
        see = SEEPolicy()
        with pytest.raises(NotImplementedError):
            see.score(entry=MagicMock(), context_vec=None)

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
        return TierManager(eat=eat, nvme_path=str(tmp_path), gpu_device=0)

    @pytest.mark.asyncio
    async def test_promote_nvme_to_ddr4(self, manager):
        with pytest.raises(NotImplementedError):
            await manager.promote(expert_id=0, shard_idx=0, target_tier=Tier.DDR4)

    @pytest.mark.asyncio
    async def test_promote_ddr4_to_vram(self, manager):
        pytest.skip("TODO Sprint 2 — richiede promozione step-by-step")

    @pytest.mark.asyncio
    async def test_evict_to_free_vram(self, manager):
        with pytest.raises(NotImplementedError):
            await manager.evict_to_free_vram(target_free_bytes=256 * 1024 * 1024)

    @pytest.mark.asyncio
    async def test_prefetch_queue(self, manager):
        with pytest.raises(NotImplementedError):
            await manager.prefetch([(0, 0), (0, 1)])


# ── Integration: EAT ↔ Tier Manager ──────────────────────────────────────────

class TestEATTierIntegration:

    @pytest.mark.asyncio
    async def test_promote_updates_eat_tier(self):
        """Post-promozione, EAT.lookup deve restituire il nuovo tier."""
        pytest.skip("TODO Sprint 2 — integration test EAT+Tier")

    @pytest.mark.asyncio
    async def test_evict_updates_eat_tier(self):
        pytest.skip("TODO Sprint 2")
