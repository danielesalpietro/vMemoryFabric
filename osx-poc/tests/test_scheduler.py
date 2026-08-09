"""Test M3 — Expert Scheduler (PT-PEP + GCSG + AER).

Coverage target: > 90%.
"""

import pytest
from scheduler import AERManager, GCSGGuard, PTPEPClassifier
from scheduler.gcsg import GatingContext

# ── PTPEPClassifier ────────────────────────────────────────────────────────────

class TestPTPEP:

    @pytest.fixture
    def ptpep_stub(self):
        """PT-PEP in stub mode (nessun modello ONNX)."""
        return PTPEPClassifier(model_path=None)

    def test_load_stub_no_error(self, ptpep_stub):
        with pytest.raises(NotImplementedError):
            ptpep_stub.load()

    def test_predict_raises_before_load(self, ptpep_stub):
        with pytest.raises(NotImplementedError):
            ptpep_stub.predict("Write a Python function to sort a list.")

    def test_predict_coding_domain(self):
        pytest.skip("TODO Sprint 3 — richiede modello ONNX fine-tuned")

    def test_predict_math_domain(self):
        pytest.skip("TODO Sprint 3")

    def test_predict_confidence_threshold(self):
        """Se confidence < threshold → DomainLabel.GENERAL come fallback."""
        pytest.skip("TODO Sprint 3")

    def test_latency_under_3ms(self):
        """Target: p99 < 3 ms su CPU (Xeon 6244 o equivalente)."""
        pytest.skip("TODO Sprint 3 — benchmark latenza PT-PEP")

    def test_hit_rate_above_70_percent(self):
        """Hit rate > 70% su 200 prompt etichettati per dominio."""
        pytest.skip("TODO Sprint 3 — richiede dataset etichettato")

    def test_predict_batch_consistent_with_single(self):
        pytest.skip("TODO Sprint 3")


# ── GCSGGuard ─────────────────────────────────────────────────────────────────

class TestGCSG:

    @pytest.fixture
    def gcsg(self):
        return GCSGGuard(theta_gate=0.85, theta_entropy=0.70, theta_contamination=0.05)

    def test_should_activate_shadow_all_conditions_met(self, gcsg):
        ctx = GatingContext(
            token_id=1,
            gating_scores=[0.9, 0.05, 0.05],
            token_entropy=0.3,
        )
        with pytest.raises(NotImplementedError):
            gcsg.should_activate_shadow(ctx)

    def test_should_not_activate_low_gating_score(self, gcsg):
        pytest.skip("TODO Sprint 3 — richiede should_activate implementato")

    def test_should_not_activate_high_entropy(self, gcsg):
        pytest.skip("TODO Sprint 3")

    def test_should_not_activate_high_contamination(self, gcsg):
        pytest.skip("TODO Sprint 3")

    def test_contamination_rate_starts_at_zero(self, gcsg):
        with pytest.raises(NotImplementedError):
            gcsg.contamination_rate()

    def test_update_thresholds(self, gcsg):
        with pytest.raises(NotImplementedError):
            gcsg.update_thresholds(theta_gate=0.90)

    def test_quality_degradation_under_2pct(self):
        """Perplexity degradazione < 2% con θ_contamination=5% — MMLU-5shot."""
        pytest.skip("TODO Sprint 3 — richiede vLLM integration + MMLU dataset")

    def test_contamination_flag_propagated_to_kvcache(self):
        pytest.skip("TODO Sprint 3 — richiede PagedAttention patch")


# ── AERManager ────────────────────────────────────────────────────────────────

class TestAER:

    def test_replication_factor_is_one_in_dev(self):
        aer = AERManager(device_ids=[0])
        assert aer.replication_factor(expert_id=0) == 1

    def test_sync_lora_delta_noop_in_dev(self):
        aer = AERManager(device_ids=[0])
        aer.sync_lora_delta(expert_id=0, delta=None)   # deve essere no-op

    def test_stats_reports_disabled(self):
        aer = AERManager(device_ids=[0])
        stats = aer.stats()
        assert stats["replication_enabled"] is False


# ── Integration: PT-PEP → Tier Manager prefetch ───────────────────────────────

class TestSchedulerTierIntegration:

    def test_ptpep_prediction_triggers_prefetch(self):
        """PT-PEP hit → prefetch_queue → TierManager.prefetch() chiamato."""
        pytest.skip("TODO Sprint 3 — integration test M3+M2")
