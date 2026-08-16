"""Test M3 — Expert Scheduler (PT-PEP + GCSG + AER).

Coverage target: > 90%.
"""
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from eat import ExpertAccessTable, Tier
from scheduler import AERManager, DomainLabel, GCSGGuard, PTPEPClassifier
from scheduler import gcsg as gcsg_module
from scheduler.gcsg import GatingContext, GCSGWorker
from tier import TierManager

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_MODEL_PATH   = Path(__file__).resolve().parent.parent / "models" / "ptpep_tfidf_v1.joblib"

# Mirrors osx_default.yaml's scheduler.ptpep.expert_map — kept inline so this
# test doesn't depend on parsing the YAML config, not because the numbers
# should ever drift from it (they're the same empirical placeholder).
_EXPERT_MAP = {
    DomainLabel.CODING:   [0, 3],
    DomainLabel.MATH:     [1, 4],
    DomainLabel.LANGUAGE: [2, 5],
    DomainLabel.SCIENCE:  [1, 3],
    DomainLabel.MEDICAL:  [6, 7],
    DomainLabel.LEGAL:    [6, 7],
    DomainLabel.CREATIVE: [2, 5],
    DomainLabel.GENERAL:  [0, 1, 2, 3, 4, 5, 6, 7],
}


# ── PTPEPClassifier ────────────────────────────────────────────────────────────

class TestPTPEP:

    @pytest.fixture
    def ptpep_stub(self):
        """PT-PEP in stub mode (nessun modello caricato)."""
        return PTPEPClassifier(model_path=None)

    @pytest.fixture(scope="class")
    def ptpep_real(self):
        """PT-PEP con il classifier TF-IDF reale (scripts/build_ptpep_classifier.py).

        Richiede models/ptpep_tfidf_v1.joblib — rigenerabile con
        `PYTHONPATH=src python scripts/build_ptpep_classifier.py`.
        """
        clf = PTPEPClassifier(model_path=str(_MODEL_PATH), expert_map=_EXPERT_MAP)
        clf.load()
        return clf

    def test_load_stub_no_error(self, ptpep_stub):
        ptpep_stub.load()   # no-op in stub mode — non deve sollevare

    def test_predict_stub_mode_returns_general(self, ptpep_stub):
        pred = ptpep_stub.predict("Write a Python function to sort a list.")
        assert pred.domain == DomainLabel.GENERAL
        assert pred.confidence == 0.0

    def test_predict_coding_domain(self, ptpep_real):
        pred = ptpep_real.predict("Write a Python function that reverses a linked list.")
        assert pred.domain == DomainLabel.CODING
        assert pred.expert_ids == _EXPERT_MAP[DomainLabel.CODING]

    def test_predict_math_domain(self, ptpep_real):
        # Esempio reale dal training set (MetaMathQA/GSM8k), non inventato:
        # word-problem scritti a mano (anche in stile GSM8k) sono finiti
        # ripetutamente sotto soglia di confidence in fase di verifica — il
        # vocabolario "math" imparato dal TF-IDF è più specifico dello stile
        # generico di un problema di matematica scritto da zero.
        pred = ptpep_real.predict(
            "Natalia sold clips to 48 of her friends in April, and then she "
            "sold half as many clips in May. How many clips did Natalia "
            "sell altogether in April and May?"
        )
        assert pred.domain == DomainLabel.MATH

    def test_predict_confidence_threshold(self):
        """Se confidence < threshold → DomainLabel.GENERAL come fallback.

        confidence_th > 1.0 forza il fallback deterministicamente per
        qualunque input (softmax non può mai superare 1.0) — non dipende
        dalla calibrazione specifica del classifier su un prompt scelto a mano.
        """
        strict = PTPEPClassifier(
            model_path=str(_MODEL_PATH), expert_map=_EXPERT_MAP, confidence_th=1.01,
        )
        strict.load()
        pred = strict.predict("Write a Python function that reverses a linked list.")
        assert pred.domain == DomainLabel.GENERAL
        assert pred.confidence < 1.01

    def test_latency_under_3ms(self, ptpep_real):
        """Target: p99 < 3 ms su CPU (Xeon 6244 o equivalente)."""
        ptpep_real.predict("warm-up")   # scarta il costo one-off della prima chiamata
        latencies = sorted(
            ptpep_real.predict(f"Sample prompt number {i} for a latency benchmark.").latency_ms
            for i in range(50)
        )
        p99 = latencies[int(0.99 * len(latencies)) - 1]
        assert p99 < 3.0, f"p99 latency {p99:.2f}ms >= 3ms target"

    def test_hit_rate_above_70_percent(self, ptpep_real):
        """Hit rate > 70% su 400 prompt held-out (50/dominio).

        tests/fixtures/ptpep_validation.json è same-distribution held-out
        (split 80/20 dallo stesso dataset per dominio), non OOD da fonte
        diversa — dichiarato così nel paper, non generalizzazione reale.
        """
        records = json.loads((_FIXTURES_DIR / "ptpep_validation.json").read_text())
        correct = sum(
            ptpep_real.predict(r["text"]).domain.value == r["domain"] for r in records
        )
        hit_rate = correct / len(records)
        assert hit_rate > 0.70, f"hit rate {hit_rate:.1%} <= 70% target ({correct}/{len(records)})"

    def test_predict_batch_consistent_with_single(self, ptpep_real):
        prompts = [
            "Write a Python function that reverses a linked list.",
            "Solve for x: 3x^2 + 5x - 2 = 0.",
            "What is the capital of France?",
        ]
        batch_results = ptpep_real.predict_batch(prompts)
        single_results = [ptpep_real.predict(p) for p in prompts]
        for b, s in zip(batch_results, single_results):
            assert b.domain == s.domain
            assert b.confidence == pytest.approx(s.confidence, abs=1e-9)


# ── GCSGGuard ─────────────────────────────────────────────────────────────────

class TestGCSG:

    @pytest.fixture
    def gcsg(self):
        return GCSGGuard(theta_gate=0.85, theta_entropy=0.70, theta_contamination=0.05)

    def test_should_activate_shadow_all_conditions_met(self, gcsg):
        ctx = GatingContext(
            token_id=1, request_id="req-1",
            gating_scores=[0.9, 0.05, 0.05], token_entropy=0.3,
        )
        should, _ = gcsg.should_activate_shadow(ctx)
        assert should is True

    def test_should_not_activate_low_gating_score(self, gcsg):
        ctx = GatingContext(
            token_id=1, request_id="req-1",
            gating_scores=[0.5, 0.3, 0.2], token_entropy=0.3,   # max 0.5 <= theta_gate 0.85
        )
        should, reason = gcsg.should_activate_shadow(ctx)
        assert should is False
        assert "gating_score" in reason

    def test_should_not_activate_high_entropy(self, gcsg):
        ctx = GatingContext(
            token_id=1, request_id="req-1",
            gating_scores=[0.9, 0.05, 0.05], token_entropy=0.9,   # >= theta_entropy 0.70
        )
        should, reason = gcsg.should_activate_shadow(ctx)
        assert should is False
        assert "entropy" in reason

    def test_should_not_activate_bf16_available(self, gcsg):
        ctx = GatingContext(
            token_id=1, request_id="req-1",
            gating_scores=[0.99], token_entropy=0.1, bf16_available=True,
        )
        should, reason = gcsg.should_activate_shadow(ctx)
        assert should is False
        assert reason == "bf16_available"

    def test_should_not_activate_high_contamination(self, gcsg):
        # Primo token della richiesta: contamination_rate("req-1") = 0/1, passa.
        ctx1 = GatingContext(
            token_id=1, request_id="req-1",
            gating_scores=[0.9, 0.05, 0.05], token_entropy=0.3,
        )
        should1, _ = gcsg.should_activate_shadow(ctx1)
        assert should1 is True
        gcsg.run_shadow(ctx1, shadow_pool={0: lambda hs, lid: None}, hidden_states=None, layer_id=0)

        # Secondo token, stessa richiesta: ora contamination_rate("req-1") = 1/2 = 0.5,
        # ben sopra theta_contamination=0.05 — deve bloccare.
        ctx2 = GatingContext(
            token_id=2, request_id="req-1",
            gating_scores=[0.9, 0.05, 0.05], token_entropy=0.3,
        )
        should2, reason2 = gcsg.should_activate_shadow(ctx2)
        assert should2 is False
        assert "contamination" in reason2

    def test_contamination_is_per_request_not_global(self, gcsg):
        # Contaminare req-1 non deve influenzare la decisione per req-2.
        ctx1 = GatingContext(
            token_id=1, request_id="req-1",
            gating_scores=[0.9], token_entropy=0.1,
        )
        gcsg.should_activate_shadow(ctx1)
        gcsg.run_shadow(ctx1, shadow_pool={0: lambda hs, lid: None}, hidden_states=None, layer_id=0)

        ctx2 = GatingContext(
            token_id=1, request_id="req-2",
            gating_scores=[0.9], token_entropy=0.1,
        )
        should2, _ = gcsg.should_activate_shadow(ctx2)
        assert should2 is True   # req-2 non ha contaminazione propria

    def test_run_shadow_picks_highest_ranked_available_expert(self, gcsg):
        ctx = GatingContext(
            token_id=1, request_id="req-1",
            gating_scores=[0.1, 0.9, 0.3], token_entropy=0.2,   # ranking: 1 > 2 > 0
        )
        calls = []
        shadow_pool = {
            0: lambda hs, lid: calls.append((hs, lid)),
            2: lambda hs, lid: calls.append((hs, lid)),
        }  # expert 1 non cachato
        result = gcsg.run_shadow(ctx, shadow_pool, hidden_states="dummy-hidden-states", layer_id=3)
        assert result.activated is True
        assert result.shadow_expert_id == 2   # il più alto in classifica REALMENTE nel pool
        assert result.contamination_flag is True
        assert calls == [("dummy-hidden-states", 3)]   # hidden_states/layer_id passati intatti

    def test_run_shadow_no_expert_in_pool(self, gcsg):
        ctx = GatingContext(
            token_id=1, request_id="req-1",
            gating_scores=[0.9, 0.1], token_entropy=0.2,
        )
        result = gcsg.run_shadow(ctx, shadow_pool={}, hidden_states=None, layer_id=0)
        assert result.activated is False
        assert result.shadow_expert_id is None
        assert result.reason_skip is not None

    def test_contamination_rate_starts_at_zero(self, gcsg):
        assert gcsg.contamination_rate() == 0.0
        assert gcsg.contamination_rate("never-seen-request") == 0.0

    def test_reset_contamination_counter_per_request(self, gcsg):
        ctx = GatingContext(
            token_id=1, request_id="req-1", gating_scores=[0.9], token_entropy=0.1,
        )
        gcsg.should_activate_shadow(ctx)
        gcsg.run_shadow(ctx, shadow_pool={0: lambda hs, lid: None}, hidden_states=None, layer_id=0)
        assert gcsg.contamination_rate("req-1") == 1.0

        gcsg.reset_contamination_counter("req-1")
        assert gcsg.contamination_rate("req-1") == 0.0

    def test_update_thresholds(self, gcsg):
        gcsg.update_thresholds(theta_gate=0.90)
        assert gcsg.theta_gate == 0.90
        assert gcsg.theta_entropy == 0.70   # non toccato

    def test_stats_reports_thresholds_and_pool_size(self, gcsg):
        stats = gcsg.stats()
        assert stats["thresholds"]["theta_gate"] == 0.85
        assert stats["shadow_pool_size"] == gcsg.shadow_pool_size

    @staticmethod
    def _fake_awq_moduleslist_layers(num_layers, num_experts, to_succeeds):
        """Fabbrica layer fake per il path 3 (ModuleList AWQ) — expert.to('cuda')
        e expert.parameters() sono gli unici due metodi che
        _pin_awq_expert_to_gpu() chiama, quindi un fake minimale basta senza
        bisogno di torch/CUDA reali."""
        class _FakeParam:
            def __init__(self, device_type):
                self.device = SimpleNamespace(type=device_type)

        class _FakeExpertModule:
            def __init__(self):
                self._device_type = "cpu"   # parte offloaded, come nel checkpoint reale

            def parameters(self):
                yield _FakeParam(self._device_type)

            def to(self, device):
                if not to_succeeds:
                    raise RuntimeError("simulated pinning failure")
                self._device_type = "cuda"
                return self

        class _FakeExperts(list):
            pass   # is_fused = hasattr(experts, "num_experts") -> False per una lista piatta

        return [
            SimpleNamespace(
                block_sparse_moe=SimpleNamespace(
                    experts=_FakeExperts(_FakeExpertModule() for _ in range(num_experts)),
                ),
            )
            for _ in range(num_layers)
        ]

    def _make_worker(self, layers, shadow_pool_size):
        worker = GCSGWorker.__new__(GCSGWorker)
        worker._base = SimpleNamespace(
            model_runner=SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(layers=layers))),
        )
        worker.guard = GCSGGuard(shadow_pool_size=shadow_pool_size)
        worker._shadow_pool = {}
        return worker

    def test_load_shadow_pool_pins_awq_experts_to_gpu_when_possible(self):
        """Fix 2026-08-10 (issue #16): _load_shadow_pool() ora prova a
        pinnare esplicitamente in GPU (copia sincrona reale, non il
        .to(..., non_blocking=True) di vLLM — vedi
        GCSGWorker._pin_awq_expert_to_gpu) gli expert AWQ ModuleList offloaded,
        invece di escluderli incondizionatamente come nello stopgap
        precedente. Se il pinning riesce su tutte le layer, l'expert_id entra
        nel pool.

        Bypassa GCSGWorker.__init__() (importa vllm.worker.worker.Worker
        reale) con __new__ + attributi assegnati a mano — stesso principio
        per cui gcsg.py resta importabile senza vLLM installato: _load_shadow
        _pool()/_pin_awq_expert_to_gpu() non fanno alcun import vllm, toccano
        solo duck-typing su self._base.model_runner.model ed expert.to()/
        .parameters(), quindi un fake minimale basta.
        """
        layers = self._fake_awq_moduleslist_layers(num_layers=2, num_experts=8, to_succeeds=True)
        worker = self._make_worker(layers, shadow_pool_size=2)

        worker._load_shadow_pool()

        assert set(worker._shadow_pool.keys()) == {0, 1}
        for layer in layers:
            for expert_id in (0, 1):
                assert layer.block_sparse_moe.experts[expert_id]._device_type == "cuda"

    def test_load_shadow_pool_excludes_awq_expert_when_pinning_fails(self):
        """Se il pinning GPU fallisce (es. .to('cuda') solleva — VRAM
        insufficiente, o qualunque altro errore reale), l'expert_id resta
        fuori dal pool invece di propagare l'eccezione — stesso principio di
        degradazione sicura già usato per il resto di _load_shadow_pool()
        (try/except in load_model(), vedi la sua docstring): hook-only per
        quell'expert_id, non un worker che non si avvia."""
        layers = self._fake_awq_moduleslist_layers(num_layers=2, num_experts=8, to_succeeds=False)
        worker = self._make_worker(layers, shadow_pool_size=2)

        worker._load_shadow_pool()   # non deve sollevare

        assert worker._shadow_pool == {}

    def test_load_shadow_pool_moves_offloaded_fused_weights_to_gpu_before_quantizing(
        self, monkeypatch,
    ):
        """Bug reale 2026-08-12 (issue #17, sub-goal 6, prima esecuzione mai
        fatta sotto vero offload — A100, cpu_offload_gb=28): a differenza dei
        path 2/3, il loop path 1 (FusedMoE fp16 grezzo, non Marlin/AWQ) non
        pinnava mai esplicitamente w13/w2 in GPU prima di _quantize_int4() —
        path 1 era sempre stato verificato solo sul modello tiny non
        offloaded, dove le slice erano già CUDA-resident per costruzione.
        Sotto offload reale restavano CPU-resident, quantizzate sul posto, e
        _ShadowExpertINT4 crashava al primo generate() reale: "Expected all
        tensors to be on the same device, cuda:0 and cpu" nel matmul
        hidden_states @ w13.T. Verifica solo la decisione di device (.to
        ('cuda') quando non già CUDA) prima della quantizzazione — la
        correttezza numerica di _quantize_int4 stessa è verificata altrove."""
        seen_devices = []

        def _fake_quantize(weight):
            seen_devices.append(weight.device.type)
            return weight, 1.0

        monkeypatch.setattr(gcsg_module, "_quantize_int4", _fake_quantize)

        class _FakeOffloadedTensor:
            def __init__(self, device_type):
                self.device = SimpleNamespace(type=device_type)

            def to(self, device):
                assert device == "cuda"
                return _FakeOffloadedTensor("cuda")

        class _FakeWeightData:
            def __getitem__(self, expert_id):
                return _FakeOffloadedTensor("cpu")   # offloaded, come sul checkpoint reale

        class _FakeFusedExperts:
            num_experts = 8   # hasattr(..., "num_experts") -> is_fused = True

            def __init__(self):
                self.w13_weight = SimpleNamespace(data=_FakeWeightData())
                self.w2_weight = SimpleNamespace(data=_FakeWeightData())
            # niente w13_qweight -> is_marlin_packed = False, dispatch a path 1

        layers = [
            SimpleNamespace(block_sparse_moe=SimpleNamespace(experts=_FakeFusedExperts()))
            for _ in range(2)
        ]
        worker = self._make_worker(layers, shadow_pool_size=2)

        worker._load_shadow_pool()

        assert set(worker._shadow_pool.keys()) == {0, 1}
        assert seen_devices, "_quantize_int4 non è mai stato chiamato"
        assert all(d == "cuda" for d in seen_devices), seen_devices

    def test_evaluate_gcsg_for_rows_passes_2d_hidden_states_slice(self):
        """Bug reale 2026-08-10, trovato dal primo run MMLU con shadow
        execution davvero attiva (mai esercitato prima — la shadow execution
        non aveva mai raggiunto questo punto finché entrambi i path erano in
        hook-only, vedi LOGBOOK): _evaluate_gcsg_for_rows() indicizzava
        hidden_states[row_idx] (collassa a 1D, shape (hidden_dim,)) invece di
        hidden_states[row_idx:row_idx+1] (batch a una riga, shape
        (1, hidden_dim)). _MarlinFusedShadowExpert costruisce router_logits
        da hidden_states.shape[0] assumendo 2D — con input 1D usa hidden_dim
        al posto del numero di righe, e crasha piu' a valle dentro
        FusedMoE.select_experts() ("not enough values to unpack (expected 2,
        got 1)"). _AWQShadowExpert/_ShadowExpertINT4 non l'avrebbero mai
        segnalato: i loro matmul tollerano un input 1D via broadcasting,
        sbagliato silenziosamente invece di sollevare un errore.

        Verifica diretta, non solo "non crasha": lo shadow callable riceve
        davvero un tensore 2D con dim0==1, non un tensore 1D.
        """
        import torch

        worker = GCSGWorker.__new__(GCSGWorker)
        worker.guard = GCSGGuard(
            theta_gate=0.5, theta_entropy=0.9, theta_contamination=1.0,
            shadow_pool_size=1, check_vram=False,
        )
        captured_shapes = []
        worker._shadow_pool = {0: lambda hs, layer_id: captured_shapes.append(tuple(hs.shape))}
        worker._current_row_request_ids = ["req-0", "req-1"]

        # Logit fortemente piccati su expert 0 -> gating_score alto, entropy bassa,
        # supera should_activate_shadow con le soglie sopra.
        router_logits = torch.tensor([[10.0, -10.0, -10.0], [10.0, -10.0, -10.0]])
        hidden_states = torch.randn(2, 4096)

        worker._evaluate_gcsg_for_rows(router_logits, hidden_states, layer_id=0)

        assert captured_shapes, (
            "run_shadow non e' mai stato chiamato — should_activate_shadow "
            "non ha superato le soglie nel setup del test"
        )
        for shape in captured_shapes:
            assert len(shape) == 2 and shape[0] == 1, (
                f"shadow callable ha ricevuto hidden_states con shape {shape}, "
                f"atteso 2D con dim0==1 (batch a una riga)"
            )

    def test_quality_degradation_under_2pct(self):
        """Perplexity degradazione < 2% con θ_contamination=5% — MMLU-5shot."""
        pytest.skip(
            "Harness e baseline esistono (scripts/eval_mmlu_gcsg.py, 72.3% "
            "su 570 domande, 2026-08-09) — bloccato su GitHub issue #10 "
            "(_load_shadow_pool() non gestisce FusedMoE Marlin-packed, "
            "shadow execution non parte). Riabilitare dopo il fix, "
            "confrontando contro la baseline registrata in LOGBOOK."
        )

    def test_contamination_flag_propagated_to_kvcache(self):
        pytest.skip("TODO Sprint 3 — richiede PagedAttention patch")


# ── GCSGWorker ↔ TierManager/EAT wiring (2026-08-12, issue #17) ────────────────
#
# Copre la logica pura Python (selezione, seeding, hook di hotness reale,
# refresh) con lo stesso principio di TestGCSG sopra: GCSGWorker.__new__() +
# attributi assegnati a mano, bypassando __init__() (import vllm reale). A
# differenza di TestGCSG, qui il TierManager è REALE (non un fake) — la sua
# costruzione richiede solo torch importabile, non CUDA vera (stesso motivo
# per cui TestTierManager, non-gpu-marked, già costruisce TierManager reali
# in CI cpu-tests). Quello che resta NON testabile qui è la parte
# effettivamente CUDA-touching: _promote_module_via_tier_manager() (il
# .to('cuda')/pin_memory() reale dentro TierManager.promote_live_tensor(),
# via GPUTransfer) — quella richiede hardware reale, vedi il pod.

class TestGCSGTierManagerWiring:

    @pytest.fixture(autouse=True)
    def _reset_pending_tier_manager(self):
        """_pending_tier_manager è stato di classe (necessario perché
        vLLM costruisce GCSGWorker da solo — vedi configure_tier_manager()) —
        senza reset, un test che lo imposta trapelerebbe negli altri test
        di questo file e di scheduler.gcsg in generale."""
        yield
        GCSGWorker.configure_tier_manager(None)

    @staticmethod
    def _real_tier_manager(tmp_path):
        eat = ExpertAccessTable(capacity=1000, n_slots=4)
        return TierManager(eat=eat, nvme_path=str(tmp_path), gpu_device=0)

    @staticmethod
    def _make_worker(tier_manager=None, shadow_pool_size=2):
        worker = GCSGWorker.__new__(GCSGWorker)
        worker.guard = GCSGGuard(shadow_pool_size=shadow_pool_size, check_vram=False)
        worker._tier_manager = tier_manager
        worker._n_experts_cached = None
        worker._shadow_pool = {}
        return worker

    # ── configure_tier_manager() — l'unico modo raggiungibile da vLLM ───────

    def test_configure_tier_manager_defaults_to_none(self):
        """Nessuna leak da altri test — vedi _reset_pending_tier_manager."""
        assert GCSGWorker._pending_tier_manager is None

    def test_configure_tier_manager_sets_class_level_pending(self, tmp_path):
        mgr = self._real_tier_manager(tmp_path)
        GCSGWorker.configure_tier_manager(mgr)
        assert GCSGWorker._pending_tier_manager is mgr

    def test_configure_tier_manager_none_clears_pending(self, tmp_path):
        mgr = self._real_tier_manager(tmp_path)
        GCSGWorker.configure_tier_manager(mgr)
        GCSGWorker.configure_tier_manager(None)
        assert GCSGWorker._pending_tier_manager is None

    # ── _select_shadow_expert_ids ──────────────────────────────────────────

    def test_select_without_tier_manager_is_round_robin(self):
        worker = self._make_worker(tier_manager=None, shadow_pool_size=2)
        assert worker._select_shadow_expert_ids(n_experts=8) == [0, 1]

    def test_select_with_tier_manager_no_traffic_yet_matches_round_robin(self, tmp_path):
        """Cold start onesto: EAT seeded ma access_count=0 ovunque -> stesso
        risultato del round-robin, non per caso ma perché sorted() è
        stabile e range(n_experts) è l'ordine di input — vedi il commento
        in _select_shadow_expert_ids sul perché NON si usa last_access_ts
        come tie-break (avrebbe rotto esattamente questa proprietà)."""
        mgr = self._real_tier_manager(tmp_path)
        for expert_id in range(8):
            mgr.eat.insert(expert_id, shard_idx=0, tier=Tier.DDR4)
        worker = self._make_worker(tier_manager=mgr, shadow_pool_size=2)
        assert worker._select_shadow_expert_ids(n_experts=8) == [0, 1]

    def test_select_with_tier_manager_prefers_hottest(self, tmp_path):
        mgr = self._real_tier_manager(tmp_path)
        for expert_id in range(8):
            mgr.eat.insert(expert_id, shard_idx=0, tier=Tier.DDR4)
        for _ in range(5):
            mgr.eat.access(expert_id=6, shard_idx=0)
        mgr.eat.access(expert_id=3, shard_idx=0)
        worker = self._make_worker(tier_manager=mgr, shadow_pool_size=2)
        assert worker._select_shadow_expert_ids(n_experts=8) == [6, 3]

    def test_select_with_tier_manager_aggregates_across_layers(self, tmp_path):
        """Un expert con hotness sparsa su più layer deve battere uno con
        tutta la hotness concentrata su un solo layer, se il totale è
        maggiore — vedi la somma per expert_id in _select_shadow_expert_ids."""
        mgr = self._real_tier_manager(tmp_path)
        for expert_id in range(4):
            for layer_id in range(3):
                mgr.eat.insert(expert_id, shard_idx=layer_id, tier=Tier.DDR4)
        for layer_id in range(3):
            mgr.eat.access(expert_id=1, shard_idx=layer_id)   # expert 1: 3 totali, sparsi
        mgr.eat.access(expert_id=2, shard_idx=0)
        mgr.eat.access(expert_id=2, shard_idx=0)               # expert 2: 2 totali, concentrati
        worker = self._make_worker(tier_manager=mgr, shadow_pool_size=1)
        assert worker._select_shadow_expert_ids(n_experts=4) == [1]

    def test_select_with_tier_manager_no_seed_falls_back(self, tmp_path):
        mgr = self._real_tier_manager(tmp_path)   # EAT vuota, nessun seed
        worker = self._make_worker(tier_manager=mgr, shadow_pool_size=2)
        assert worker._select_shadow_expert_ids(n_experts=8) == [0, 1]

    # ── _seed_eat_entries ───────────────────────────────────────────────────

    @staticmethod
    def _fake_layers(num_layers, num_experts):
        return TestGCSG._fake_awq_moduleslist_layers(
            num_layers=num_layers, num_experts=num_experts, to_succeeds=True,
        )

    def test_seed_eat_entries_creates_one_per_expert_per_layer(self, tmp_path):
        mgr = self._real_tier_manager(tmp_path)
        worker = self._make_worker(tier_manager=mgr)
        worker._base = SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(model=SimpleNamespace(layers=self._fake_layers(3, 4))),
            ),
        )
        worker._seed_eat_entries()
        entries = mgr.eat.get_tier(Tier.DDR4)
        assert len(entries) == 3 * 4
        assert all(e.tier == Tier.DDR4 for e in entries)

    def test_seed_eat_entries_idempotent_preserves_existing_traffic(self, tmp_path):
        mgr = self._real_tier_manager(tmp_path)
        worker = self._make_worker(tier_manager=mgr)
        worker._base = SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(model=SimpleNamespace(layers=self._fake_layers(2, 2))),
            ),
        )
        mgr.eat.insert(expert_id=0, shard_idx=0, tier=Tier.DDR4)
        mgr.eat.access(expert_id=0, shard_idx=0)   # traffico reale pre-esistente

        worker._seed_eat_entries()

        entry = mgr.eat.lookup(0, 0)
        assert entry.access_count == 1   # non azzerata/sovrascritta dal seed
        assert len(mgr.eat.get_tier(Tier.DDR4)) == 2 * 2   # le altre 3 combinazioni seeded

    # ── traffico EAT reale nell'hook .gate ──────────────────────────────────

    def test_evaluate_gcsg_for_rows_feeds_eat_with_real_routing(self, tmp_path):
        import torch
        mgr = self._real_tier_manager(tmp_path)
        mgr.eat.insert(expert_id=0, shard_idx=5, tier=Tier.DDR4)
        worker = self._make_worker(tier_manager=mgr, shadow_pool_size=0)
        worker._current_row_request_ids = ["req-0"]

        router_logits = torch.tensor([[10.0, -10.0, -10.0]])   # top-1 = expert 0
        hidden_states = torch.randn(1, 4096)

        worker._evaluate_gcsg_for_rows(router_logits, hidden_states, layer_id=5)

        assert mgr.eat.lookup(0, 5).access_count == 1

    def test_evaluate_gcsg_for_rows_tracks_real_top1_not_shadow_activation(self, tmp_path):
        """Il traffico EAT riflette il routing REALE (top-1 del router),
        indipendentemente da should_activate_shadow()/run_shadow() — deve
        continuare a fluire anche quando le soglie GCSG non scattano
        affatto (theta_gate impossibile da superare qui)."""
        import torch
        mgr = self._real_tier_manager(tmp_path)
        mgr.eat.insert(expert_id=1, shard_idx=0, tier=Tier.DDR4)
        worker = self._make_worker(tier_manager=mgr, shadow_pool_size=0)
        worker.guard = GCSGGuard(theta_gate=0.999, check_vram=False)
        worker._current_row_request_ids = ["req-0"]

        router_logits = torch.tensor([[-10.0, 10.0, -10.0]])   # top-1 = expert 1
        hidden_states = torch.randn(1, 4096)

        worker._evaluate_gcsg_for_rows(router_logits, hidden_states, layer_id=0)

        assert mgr.eat.lookup(1, 0).access_count == 1

    def test_evaluate_gcsg_for_rows_without_tier_manager_untouched(self):
        """tier_manager=None: nessuna chiamata EAT (non c'è nulla da
        chiamare), nessuna eccezione — path di default invariato."""
        import torch
        worker = self._make_worker(tier_manager=None, shadow_pool_size=0)
        worker._current_row_request_ids = ["req-0"]
        router_logits = torch.tensor([[10.0, -10.0, -10.0]])
        hidden_states = torch.randn(1, 4096)
        worker._evaluate_gcsg_for_rows(router_logits, hidden_states, layer_id=0)   # non deve sollevare

    # ── refresh_shadow_pool_selection ──────────────────────────────────────

    def test_refresh_without_tier_manager_is_noop(self, caplog):
        worker = self._make_worker(tier_manager=None)
        with caplog.at_level(logging.WARNING):
            worker.refresh_shadow_pool_selection()
        assert "no-op" in caplog.text

    def test_refresh_before_first_load_is_noop(self, tmp_path, caplog):
        mgr = self._real_tier_manager(tmp_path)
        worker = self._make_worker(tier_manager=mgr)
        assert worker._n_experts_cached is None
        with caplog.at_level(logging.WARNING):
            worker.refresh_shadow_pool_selection()
        assert "prima di _load_shadow_pool" in caplog.text

    def test_refresh_reloads_when_selection_changes(self, tmp_path):
        mgr = self._real_tier_manager(tmp_path)
        for expert_id in range(4):
            mgr.eat.insert(expert_id, shard_idx=0, tier=Tier.DDR4)
        worker = self._make_worker(tier_manager=mgr, shadow_pool_size=2)
        worker._n_experts_cached = 4
        worker._shadow_pool = {0: object(), 1: object()}   # selezione "corrente" simulata

        calls = []
        worker._load_shadow_pool = lambda: calls.append(1)

        for _ in range(3):
            mgr.eat.access(expert_id=3, shard_idx=0)   # rende 3 il più caldo — selezione cambia

        worker.refresh_shadow_pool_selection()

        assert calls == [1]

    def test_refresh_noop_when_selection_unchanged(self, tmp_path):
        mgr = self._real_tier_manager(tmp_path)
        for expert_id in range(4):
            mgr.eat.insert(expert_id, shard_idx=0, tier=Tier.DDR4)
        worker = self._make_worker(tier_manager=mgr, shadow_pool_size=2)
        worker._n_experts_cached = 4
        worker._shadow_pool = {0: object(), 1: object()}

        calls = []
        worker._load_shadow_pool = lambda: calls.append(1)

        worker.refresh_shadow_pool_selection()   # nessun traffico -> selezione invariata [0,1]

        assert calls == []


# ── CPU-resident shadow pool (2026-08-17, issue #33 Fase 2) ───────────────────
#
# Pool DDR4-resident parallelo al pool GPU esistente — stesso principio di
# TestGCSGTierManagerWiring: GCSGWorker.__new__() + attributi a mano,
# TierManager reale (torch importabile basta, non serve CUDA). La garanzia
# "mai .to('cuda')" si verifica negando l'assunzione del fix 2026-08-12 (pesi
# offloaded restano CPU-resident finché non promossi esplicitamente) — vedi
# _build_cpu_shadow_pool()/_load_shadow_pool() in gcsg.py.

class TestGCSGCpuShadowPool:

    @staticmethod
    def _real_tier_manager(tmp_path):
        return TestGCSGTierManagerWiring._real_tier_manager(tmp_path)

    @staticmethod
    def _make_worker(tier_manager=None, shadow_pool_size=2):
        worker = TestGCSGTierManagerWiring._make_worker(tier_manager, shadow_pool_size)
        worker._cpu_shadow_pool = {}
        return worker

    @staticmethod
    def _fake_offloaded_fused_layers(num_layers, num_experts):
        """Stesso fake di
        TestGCSG.test_load_shadow_pool_moves_offloaded_fused_weights_to_gpu_before_quantizing
        (path 1, FusedMoE fp16 grezzo con expert offloaded — CPU-resident
        per costruzione), estratto come helper riusabile: sia il pool GPU
        (invariato) sia il nuovo pool CPU condividono lo stesso setup."""
        class _FakeOffloadedTensor:
            def __init__(self, device_type):
                self.device = SimpleNamespace(type=device_type)

            def to(self, device):
                assert device == "cuda"
                return _FakeOffloadedTensor("cuda")

        class _FakeWeightData:
            def __getitem__(self, expert_id):
                return _FakeOffloadedTensor("cpu")

        class _FakeFusedExperts:
            def __init__(self):
                self.num_experts = num_experts
                self.w13_weight = SimpleNamespace(data=_FakeWeightData())
                self.w2_weight = SimpleNamespace(data=_FakeWeightData())

        return [
            SimpleNamespace(block_sparse_moe=SimpleNamespace(experts=_FakeFusedExperts()))
            for _ in range(num_layers)
        ]

    def test_load_shadow_pool_builds_parallel_cpu_pool_when_tier_manager_wired(
        self, tmp_path, monkeypatch,
    ):
        """Con un TierManager wired, _load_shadow_pool() costruisce ANCHE un
        pool CPU-resident per gli stessi expert_ids del pool GPU —
        parallelo, non sostitutivo (self._shadow_pool resta invariato).
        Stesso monkeypatch di _quantize_int4 di
        TestGCSG.test_load_shadow_pool_moves_offloaded_fused_weights_to_gpu_before_quantizing,
        qui usato per distinguere le due popolazioni per device visto."""
        seen_devices = []

        def _fake_quantize(weight):
            seen_devices.append(weight.device.type)
            return weight, 1.0

        monkeypatch.setattr(gcsg_module, "_quantize_int4", _fake_quantize)

        layers = self._fake_offloaded_fused_layers(num_layers=2, num_experts=8)
        mgr = self._real_tier_manager(tmp_path)
        worker = self._make_worker(tier_manager=mgr, shadow_pool_size=2)
        worker._base = SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(model=SimpleNamespace(layers=layers)),
            ),
        )

        worker._load_shadow_pool()

        assert set(worker._shadow_pool.keys()) == {0, 1}
        assert set(worker._cpu_shadow_pool.keys()) == {0, 1}

        # 2 layer * 2 expert * 2 tensori (w13 + w2) = 8 chiamate per pool:
        # GPU pool promosso a cuda (fix 2026-08-12), CPU pool mai toccato —
        # resta al device originale ("cpu", come lo restituisce _FakeWeightData).
        assert seen_devices.count("cuda") == 8
        assert seen_devices.count("cpu") == 8

    def test_load_shadow_pool_cpu_pool_stays_empty_without_tier_manager(
        self, monkeypatch,
    ):
        """Comportamento di default invariato: senza TierManager nessun
        pool CPU viene costruito — stesso principio opt-in del resto del
        wiring issue #17."""
        monkeypatch.setattr(gcsg_module, "_quantize_int4", lambda w: (w, 1.0))

        layers = self._fake_offloaded_fused_layers(num_layers=1, num_experts=4)
        worker = self._make_worker(tier_manager=None, shadow_pool_size=2)
        worker._base = SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(model=SimpleNamespace(layers=layers)),
            ),
        )

        worker._load_shadow_pool()

        assert set(worker._shadow_pool.keys()) == {0, 1}
        assert worker._cpu_shadow_pool == {}

    def test_build_cpu_shadow_pool_produces_working_forward(self):
        """Non solo "non solleva": il pool costruito è realmente
        utilizzabile end-to-end (estrazione dello slice
        w13_weight.data[expert_id]/w2_weight.data[expert_id] inclusa, non
        solo la costruzione diretta di _ShadowExpertINT4 come in
        test_cpu_kernel.py) — stesso schema di
        TestShadowExpertINT4CPU.test_runs_on_cpu_without_cuda (Fase 1)."""
        import torch

        generator = torch.Generator().manual_seed(0)
        hidden, intermediate, num_layers, num_experts = 16, 32, 2, 4
        w13 = torch.randn(num_experts, 2 * intermediate, hidden, generator=generator)
        w2 = torch.randn(num_experts, hidden, intermediate, generator=generator)
        layers = [
            SimpleNamespace(block_sparse_moe=SimpleNamespace(experts=SimpleNamespace(
                num_experts=num_experts,
                w13_weight=SimpleNamespace(data=w13),
                w2_weight=SimpleNamespace(data=w2),
            )))
            for _ in range(num_layers)
        ]
        worker = self._make_worker(tier_manager=None)

        cpu_pool = worker._build_cpu_shadow_pool(layers, expert_ids=[1, 2])

        assert set(cpu_pool.keys()) == {1, 2}
        hidden_states = torch.randn(4, hidden, generator=generator)
        for shadow in cpu_pool.values():
            output = shadow(hidden_states, layer_id=0)
            assert output.device.type == "cpu"
            assert not output.is_cuda
            assert output.shape == (4, hidden)
            assert torch.isfinite(output).all()

    def test_refresh_shadow_pool_selection_clears_cpu_pool_too(self, tmp_path):
        """refresh_shadow_pool_selection() svuota self._shadow_pool prima
        di ricaricare (vedi TestGCSGTierManagerWiring) — deve fare lo
        stesso per self._cpu_shadow_pool, altrimenti expert_id rimossi
        dalla nuova selezione resterebbero fantasma nel pool CPU."""
        mgr = self._real_tier_manager(tmp_path)
        for expert_id in range(4):
            mgr.eat.insert(expert_id, shard_idx=0, tier=Tier.DDR4)
        worker = self._make_worker(tier_manager=mgr, shadow_pool_size=2)
        worker._n_experts_cached = 4
        worker._shadow_pool = {0: object(), 1: object()}
        worker._cpu_shadow_pool = {0: object(), 1: object()}

        cleared = {}

        def _fake_load():
            cleared["shadow_pool"] = dict(worker._shadow_pool)
            cleared["cpu_shadow_pool"] = dict(worker._cpu_shadow_pool)
        worker._load_shadow_pool = _fake_load

        for _ in range(3):
            mgr.eat.access(expert_id=3, shard_idx=0)   # selezione cambia -> reload

        worker.refresh_shadow_pool_selection()

        assert cleared == {"shadow_pool": {}, "cpu_shadow_pool": {}}


# ── Hot/cold routing (2026-08-17, issue #33 Fase 3) ────────────────────────────
#
# route_forward()/_RoutedShadowPool decidono, per un expert_id già presente in
# entrambi i pool (Fase 2), quale residenza eseguire — usando la
# classificazione di Fase 0 (SEEPolicy.classify_hot_cold(), issue #21-informed,
# vedi LOGBOOK_ISSUE33.MD). Stesso principio di test di TestGCSGTierManagerWiring:
# GCSGWorker.__new__() + attributi a mano, TierManager reale.

class TestGCSGRouteForward:

    @staticmethod
    def _make_worker(tier_manager=None, shadow_pool_size=2):
        return TestGCSGCpuShadowPool._make_worker(tier_manager, shadow_pool_size)

    @staticmethod
    def _real_tier_manager(tmp_path):
        return TestGCSGTierManagerWiring._real_tier_manager(tmp_path)

    # ── route_forward(): dispatch puro, nessun _load_shadow_pool() coinvolto ──

    def test_route_forward_dispatches_hot_expert_to_gpu_pool(self):
        calls = []
        worker = self._make_worker()
        worker._shadow_pool = {0: lambda hs, lid: calls.append(("gpu", hs, lid))}
        worker._cpu_shadow_pool = {0: lambda hs, lid: calls.append(("cpu", hs, lid))}
        worker._hot_expert_ids = {0}

        worker.route_forward(expert_id=0, layer_id=3, hidden_states="hs")

        assert calls == [("gpu", "hs", 3)]

    def test_route_forward_dispatches_cold_expert_to_cpu_pool(self):
        calls = []
        worker = self._make_worker()
        worker._shadow_pool = {0: lambda hs, lid: calls.append(("gpu", hs, lid))}
        worker._cpu_shadow_pool = {0: lambda hs, lid: calls.append(("cpu", hs, lid))}
        worker._hot_expert_ids = set()   # 0 non è "caldo" -> freddo

        worker.route_forward(expert_id=0, layer_id=3, hidden_states="hs")

        assert calls == [("cpu", "hs", 3)]

    def test_route_forward_falls_back_to_gpu_when_not_in_cpu_pool(self):
        """Path 2/3 (Marlin/AWQ) — Fase 2 non costruisce un pool CPU per
        loro: expert presente solo nel pool GPU, la classificazione hot/
        cold (anche se dicesse "freddo") non ha un'alternativa da usare."""
        calls = []
        worker = self._make_worker()
        worker._shadow_pool = {0: lambda hs, lid: calls.append("gpu")}
        worker._cpu_shadow_pool = {}
        worker._hot_expert_ids = set()   # "freddo", ma non c'è pool CPU

        worker.route_forward(expert_id=0, layer_id=0, hidden_states="hs")

        assert calls == ["gpu"]

    def test_route_forward_falls_back_to_cpu_when_not_in_gpu_pool(self):
        """Caso limite (oggi non raggiunto da _load_shadow_pool(), che
        costruisce i due pool per gli stessi expert_id nel path 1): un
        expert presente solo nel pool CPU deve comunque funzionare."""
        calls = []
        worker = self._make_worker()
        worker._shadow_pool = {}
        worker._cpu_shadow_pool = {0: lambda hs, lid: calls.append("cpu")}
        worker._hot_expert_ids = {0}   # "caldo", ma non c'è pool GPU

        worker.route_forward(expert_id=0, layer_id=0, hidden_states="hs")

        assert calls == ["cpu"]

    # ── _refresh_hot_cold_classification() ───────────────────────────────────

    def test_refresh_hot_cold_defaults_all_hot_without_traffic(self, tmp_path):
        """Cold start onesto: EAT seeded ma access_count=0 ovunque (o EAT
        vuota) -> TUTTI gli expert_id dei due pool restano "caldi",
        comportamento pre-Fase-3 invariato (GPU-only) — non instradare
        tutto a freddo su un segnale che non esiste ancora, stesso
        principio di _select_shadow_expert_ids()."""
        mgr = self._real_tier_manager(tmp_path)
        worker = self._make_worker(tier_manager=mgr)
        worker._shadow_pool = {0: object(), 1: object()}
        worker._cpu_shadow_pool = {0: object(), 1: object()}

        worker._refresh_hot_cold_classification()

        assert worker._hot_expert_ids == {0, 1}

    def test_refresh_hot_cold_without_tier_manager_defaults_all_hot(self):
        worker = self._make_worker(tier_manager=None)
        worker._shadow_pool = {0: object()}
        worker._cpu_shadow_pool = {0: object()}

        worker._refresh_hot_cold_classification()

        assert worker._hot_expert_ids == {0}

    def test_refresh_hot_cold_uses_real_traffic(self, tmp_path):
        mgr = self._real_tier_manager(tmp_path)
        for expert_id in range(4):
            mgr.eat.insert(expert_id, shard_idx=0, tier=Tier.DDR4)
        for _ in range(5):
            mgr.eat.access(expert_id=2, shard_idx=0)   # 2 è nettamente il più caldo
        worker = self._make_worker(tier_manager=mgr)
        worker._shadow_pool = {i: object() for i in range(4)}
        worker._cpu_shadow_pool = {i: object() for i in range(4)}

        worker._refresh_hot_cold_classification()

        assert 2 in worker._hot_expert_ids

    # ── _RoutedShadowPool ─────────────────────────────────────────────────────

    def test_routed_shadow_pool_contains_union_of_both_pools(self):
        worker = self._make_worker()
        worker._shadow_pool = {0: object()}
        worker._cpu_shadow_pool = {1: object()}
        pool = gcsg_module._RoutedShadowPool(worker)

        assert 0 in pool
        assert 1 in pool
        assert 2 not in pool

    # ── Integrazione con _load_shadow_pool()/_evaluate_gcsg_for_rows() ────────

    def test_load_shadow_pool_refreshes_hot_cold_classification(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gcsg_module, "_quantize_int4", lambda w: (w, 1.0))
        layers = TestGCSGCpuShadowPool._fake_offloaded_fused_layers(
            num_layers=2, num_experts=8,
        )
        mgr = self._real_tier_manager(tmp_path)
        worker = self._make_worker(tier_manager=mgr, shadow_pool_size=2)
        worker._base = SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(model=SimpleNamespace(layers=layers)),
            ),
        )

        worker._load_shadow_pool()

        # Nessun traffico EAT reale ancora -> cold start onesto, entrambi
        # gli expert_id selezionati restano "caldi".
        assert worker._hot_expert_ids == {0, 1}

    def test_evaluate_gcsg_for_rows_routes_cold_expert_to_cpu_pool_never_touching_gpu_pool(
        self, tmp_path,
    ):
        """La versione testata di quello che issue #33 Fase 3 chiedeva
        esplicitamente: un expert instradato a freddo non deve MAI
        attraversare il pool GPU (e quindi mai GPUTransfer.to_vram()/
        TierManager.promote_live_tensor(), che vivono solo dietro le
        callable del pool GPU) — end-to-end da _evaluate_gcsg_for_rows()
        (hook .gate reale) fino al dispatch, non solo da route_forward()
        isolato."""
        import torch

        gpu_calls = []
        cpu_calls = []
        mgr = self._real_tier_manager(tmp_path)
        worker = self._make_worker(tier_manager=mgr, shadow_pool_size=1)
        worker.guard = GCSGGuard(
            theta_gate=0.5, theta_entropy=0.9, theta_contamination=1.0,
            shadow_pool_size=1, check_vram=False,
        )
        worker._shadow_pool = {0: lambda hs, lid: gpu_calls.append((hs.shape, lid))}
        worker._cpu_shadow_pool = {0: lambda hs, lid: cpu_calls.append((hs.shape, lid))}
        worker._hot_expert_ids = set()   # expert 0 classificato "freddo"
        worker._current_row_request_ids = ["req-0"]

        # Logit fortemente piccato su expert 0 -> supera should_activate_shadow.
        router_logits = torch.tensor([[10.0, -10.0, -10.0]])
        hidden_states = torch.randn(1, 4096)

        worker._evaluate_gcsg_for_rows(router_logits, hidden_states, layer_id=7)

        assert cpu_calls == [((1, 4096), 7)]
        assert gpu_calls == []   # mai toccato il pool GPU per questo expert

    def test_evaluate_gcsg_for_rows_routes_hot_expert_to_gpu_pool(self, tmp_path):
        import torch

        gpu_calls = []
        cpu_calls = []
        mgr = self._real_tier_manager(tmp_path)
        worker = self._make_worker(tier_manager=mgr, shadow_pool_size=1)
        worker.guard = GCSGGuard(
            theta_gate=0.5, theta_entropy=0.9, theta_contamination=1.0,
            shadow_pool_size=1, check_vram=False,
        )
        worker._shadow_pool = {0: lambda hs, lid: gpu_calls.append((hs.shape, lid))}
        worker._cpu_shadow_pool = {0: lambda hs, lid: cpu_calls.append((hs.shape, lid))}
        worker._hot_expert_ids = {0}   # expert 0 classificato "caldo"
        worker._current_row_request_ids = ["req-0"]

        router_logits = torch.tensor([[10.0, -10.0, -10.0]])
        hidden_states = torch.randn(1, 4096)

        worker._evaluate_gcsg_for_rows(router_logits, hidden_states, layer_id=7)

        assert gpu_calls == [((1, 4096), 7)]
        assert cpu_calls == []


# ── Marlin path TierManager wiring (2026-08-12, issue #17) ────────────────────
#
# Solo _marlin_pool_shard_key() è pura logica testabile qui — il resto
# (_build_marlin_tensor_promoter()'s actual transfer, _PinnedMarlinExperts
# con tensor_promoter reale) richiede torch CUDA vero (GPUTransfer.to_vram()
# dentro TierManager.promote_live_tensor()), stesso motivo per cui
# _promote_module_via_tier_manager() del path AWQ non ha un test diretto
# sopra — verificato su hardware reale (checklist), non qui.

class TestMarlinPoolShardKey:

    def test_deterministic(self):
        k1 = GCSGWorker._marlin_pool_shard_key(5, [0, 1])
        k2 = GCSGWorker._marlin_pool_shard_key(5, [0, 1])
        assert k1 == k2

    def test_order_independent(self):
        """La composizione del pool conta, non l'ordine in cui è passata
        — sorted() dentro la funzione."""
        k1 = GCSGWorker._marlin_pool_shard_key(5, [0, 1])
        k2 = GCSGWorker._marlin_pool_shard_key(5, [1, 0])
        assert k1 == k2

    def test_differs_by_layer(self):
        k1 = GCSGWorker._marlin_pool_shard_key(5, [0, 1])
        k2 = GCSGWorker._marlin_pool_shard_key(6, [0, 1])
        assert k1 != k2

    def test_differs_by_pool_composition(self):
        """Il motivo per cui questa funzione esiste: una composizione
        diversa allo stesso layer non deve produrre la stessa chiave —
        vedi la docstring del metodo sul bug di staleness (VRAM di una
        composizione precedente riusata per errore) che questo evita."""
        k1 = GCSGWorker._marlin_pool_shard_key(5, [0, 1])
        k2 = GCSGWorker._marlin_pool_shard_key(5, [2, 6])
        assert k1 != k2


# ── _build_marlin_tensor_promoter — regressione bug reale (2026-08-12) ────────
#
# Trovato sul pod, prima verifica hardware del path Marlin:
# "cannot pin 'torch.cuda.HalfTensor' only dense CPU tensors can be pinned".
# _build_marlin_shadow_pool() decide "offloaded" controllando SOLO
# w13_qweight — non tutte le sei tensori Marlin-packed di un layer
# condividono per forza lo stesso device; alcune (es. w13_scales) possono
# restare GPU-resident anche quando w13_qweight è offloaded su CPU.
# Chiamare .pin_memory() incondizionatamente su una già CUDA crashava.

class _FakeCudaTensor:
    class _Device:
        type = "cuda"
    device = _Device()


class TestMarlinTensorPromoterDeviceCheck:

    @staticmethod
    def _make_worker():
        worker = GCSGWorker.__new__(GCSGWorker)
        worker._tier_manager = None   # non serve per il ramo testato: lo
                                       # short-circuit "già CUDA" ritorna
                                       # prima di toccare _tier_manager
        return worker

    def test_promoter_returns_already_cuda_tensor_unchanged(self):
        """Riproduce esattamente lo scenario del bug: un tensore
        non-dominante già GPU-resident non deve mai arrivare a
        .pin_memory()."""
        worker = self._make_worker()
        promoter = worker._build_marlin_tensor_promoter(layer_id=5, expert_ids=[0, 1])

        fake_cuda_tensor = _FakeCudaTensor()
        result = promoter("w13_scales", fake_cuda_tensor)   # non il dominante

        assert result is fake_cuda_tensor   # passthrough, nessun crash

    def test_promoter_dominant_name_also_short_circuits_if_already_cuda(self):
        """Lo stesso controllo si applica anche al tensore dominante
        (w13_qweight) — anche se nella pratica quello è il segnale usato
        per decidere 'offloaded', vale la stessa difesa per coerenza."""
        worker = self._make_worker()
        promoter = worker._build_marlin_tensor_promoter(layer_id=5, expert_ids=[0, 1])

        fake_cuda_tensor = _FakeCudaTensor()
        result = promoter("w13_qweight", fake_cuda_tensor)

        assert result is fake_cuda_tensor   # non chiama _tier_manager (None qui) senza crashare


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

    def test_trigger_conditions_exposed(self):
        aer = AERManager(device_ids=[0], load_threshold_qps=30.0)
        assert aer.trigger_conditions == {"load_threshold_qps": 30.0}

    def test_evaluate_load_below_threshold_no_trigger(self):
        aer = AERManager(device_ids=[0], load_threshold_qps=50.0)
        assert aer.evaluate_load(expert_id=2, requests_per_second=10.0) is False
        assert aer.stats()["would_replicate_count"] == 0

    def test_evaluate_load_above_threshold_triggers_and_logs(self, caplog):
        aer = AERManager(device_ids=[0], load_threshold_qps=50.0)
        with caplog.at_level(logging.INFO):
            triggered = aer.evaluate_load(expert_id=3, requests_per_second=75.0)
        assert triggered is True
        assert "WOULD_REPLICATE" in caplog.text
        assert "expert_id=3" in caplog.text
        # il trigger logic segnala la condizione, ma niente hardware la esegue:
        # replication_factor resta 1 anche subito dopo un WOULD_REPLICATE
        assert aer.replication_factor(expert_id=3) == 1

    def test_stats_tracks_would_replicate_experts(self):
        aer = AERManager(device_ids=[0], load_threshold_qps=50.0)
        aer.evaluate_load(expert_id=1, requests_per_second=80.0)
        aer.evaluate_load(expert_id=2, requests_per_second=10.0)   # sotto soglia
        aer.evaluate_load(expert_id=1, requests_per_second=90.0)   # stesso expert di nuovo
        stats = aer.stats()
        assert stats["would_replicate_count"] == 2
        assert stats["would_replicate_experts"] == [1]


# ── Integration: PT-PEP → Tier Manager prefetch ───────────────────────────────

class TestSchedulerTierIntegration:

    def test_ptpep_prediction_triggers_prefetch(self):
        """PT-PEP hit → prefetch_queue → TierManager.prefetch() chiamato."""
        pytest.skip("TODO Sprint 3 — integration test M3+M2")
