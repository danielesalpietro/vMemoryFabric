"""M3 — GCSG: Gating Confidence Shadow Guard.

Intercetta i gating scores post-router MoE e attiva una shadow execution
con expert INT4 quando la confidenza è alta e la contaminazione è bassa.

Parametri default (calibrabili a runtime):
    θ_gate          = 0.85   (gating score minimo per shadow)
    θ_entropy       = 0.70   (token entropy massima per shadow)
    θ_contamination = 0.05   (tasso contaminazione KV-Cache massimo)

Shadow execution attivata solo se TUTTE le condizioni:
    - BF16 non disponibile (o budget VRAM insufficiente) — segnalato dal
      chiamante via GatingContext.bf16_available, GCSGGuard non lo deduce
      da solo (dipenderebbe da Tier Manager/EAT, integrazione M2+M3 deferred,
      vedi TestSchedulerTierIntegration)
    - gating_score  > θ_gate
    - token_entropy < θ_entropy
    - contamination < θ_contamination (per request_id, non globale — vedi sotto)

Vincolo dev (single GPU 3090 24 GB):
    Memory math (Mixtral 8x7B, verificata 2026-08-08): un "expert" INT4 qui
    significa l'FFN con indice i su tutti e 32 i layer (il routing MoE è
    per-layer, non c'è un unico blocco "expert i"). Per layer: 3 proiezioni
    SwiGLU × 4096×14336 ≈ 176M parametri; × 32 layer ≈ 5.6B parametri/expert;
    a INT4 (~0.5 B/param + overhead scale) ≈ 2.8–3.2 GB, quindi ~3 GB/expert.

    Con il modello attivo a ~14 GB (lower bound, scheduler/__init__.py):
    restano ~10 GB → max 3 shadow expert prima di intaccare il budget
    KV-cache di vLLM. A ~16 GB (upper bound): restano ~8 GB → max 2.
    shadow_pool_size in config è quindi 2, non 4 — top-4 supererebbe i 24 GB
    in ogni scenario realistico, anche solo col modello + shadow pool, prima
    ancora di contare KV-cache e overhead CUDA context.

    Il preflight in GCSGGuard.__init__ non si limita a un fail-fast binario:
    se il modello ha consumato più di ~18 GB all'avvio (restano <6 GB, non
    bastano 2 expert × 3 GB), abbassa shadow_pool_size effettivo a 1 e
    prosegue — GCSG degradato ma operativo. Fallisce fast solo se non entra
    nemmeno 1 expert (altrimenti l'OOM arriverebbe a runtime sotto carico,
    molto peggio da diagnosticare).

Hook vLLM — verificato contro il sorgente reale di vllm==0.6.6.post1
(Sprint 3, 2026-08-08), non assunto:

    1. NON un monkey-patch su _run_workers(). In vLLM 0.6.x ModelRunner gira
       in worker process separati (multiprocessing/Ray) — un patch sul
       processo principale non si propaga ai worker. L'hook corretto è una
       sottoclasse GCSGWorker(Worker) passata a LLMEngine via
       EngineArgs(worker_cls="scheduler.gcsg.GCSGWorker") — STRINGA
       qualname, non la classe: verificato (smoke test 2026-08-09) che
       vllm.worker.worker_base.init_worker() fa
       resolve_obj_by_qualname(qualname).rsplit(".", 1), cioè importa e
       risolve da solo un percorso "modulo.Classe" — passare la classe
       GCSGWorker direttamente crash con
       AttributeError: type object 'GCSGWorker' has no attribute 'rsplit'.

    2. Shadow pool: va caricato in GCSGWorker.load_model(), DOPO aver
       chiamato super().load_model() — non in init_device(). Verificato in
       vllm.executor.gpu_executor.GPUExecutor: la sequenza di avvio è
       `driver_worker.init_device()` seguito da `driver_worker.load_model()`.
       init_device() gira per PRIMO e già lì cattura una baseline di VRAM
       libera (self.init_gpu_memory = torch.cuda.mem_get_info()[0]) PRIMA
       che il modello sia caricato — se lo shadow pool venisse allocato lì,
       il preflight VRAM di GCSGGuard sovrastimerebbe lo spazio disponibile
       esattamente come temuto. load_model() invece chiama
       self.model_runner.load_model() e poi ritorna — lo shadow pool va
       aggiunto subito dopo quella chiamata, quando la VRAM del modello
       principale è già stata allocata per davvero.

    3. Gating scores: NON leggibili dall'output di ModelRunner.execute_model()
       — quel metodo ritorna SamplerOutput (token campionati), non gating
       scores. Il router MoE è vllm.model_executor.models.mixtral.MixtralMoE
       (non "MixtralSparseMoeBlock", nome che non esiste in questa versione).
       MixtralMoE.forward() calcola `router_logits, _ = self.gate(hidden_states)`
       ma NON lo restituisce — router_logits resta una variabile locale,
       scartata dopo l'uso. self.gate è un ReplicatedLinear la cui forward()
       ritorna (output, output_bias): output QUI è router_logits, shape
       (num_tokens, num_experts). L'hook corretto è quindi un forward hook
       PyTorch su ogni layer_i.block_sparse_moe.gate (non sul blocco MoE
       intero, il cui hook vedrebbe solo l'output finale già mescolato dagli
       expert) — registrato in GCSGWorker.load_model() dopo il caricamento,
       iterando self.model_runner.model.model.layers[i].block_sparse_moe.gate.

    4. Contaminazione per-request, non globale — CORRETTO durante lo smoke
       test (2026-08-09), prima ipotesi sbagliata: GCSGWorker.execute_model()
       sovrascrive WorkerBase.execute_model(execute_model_req:
       ExecuteModelRequest), NON ModelRunner.execute_model(model_input, ...)
       — stesso nome, parametro diverso. ExecuteModelRequest non ha un campo
       request_ids_to_seq_ids (verificato: assente da dir()); i request_id
       reali si leggono da execute_model_req.seq_group_metadata_list
       (List[SequenceGroupMetadata], ognuno con .request_id), confermato con
       un run reale via logging di debug prima di correggere il codice.
       GatingContext.request_id e GCSGGuard.contamination_rate(request_id)
       riflettono questo — solo il punto di estrazione era sbagliato.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pynvml  # provided by the `nvidia-ml-py` package (requirements.txt) —

# the standalone `pynvml` PyPI package is deprecated, this
# is the same import name from NVIDIA's maintained bindings
# M1/M2 — pure-Python project packages, no vLLM/CUDA hard dependency at
# import time (tier/gpu.py's GPUTransfer only requires torch at
# *construction*, and CI cpu-tests already imports both packages directly —
# see tests/test_tier.py, tests/test_eat.py). Safe at module scope, unlike
# the local vllm imports below (see docstring above).
from eat import Tier
from tier import SEEPolicy, TierManager

log = logging.getLogger(__name__)


@dataclass
class GatingContext:
    """Contesto gating per un singolo token/step.

    request_id lega il token alla sua sequenza — necessario perché vLLM
    processa più richieste in parallelo nello stesso batch: senza, la
    contaminazione di richieste diverse finirebbe in un unico contatore
    globale e θ_contamination diventerebbe inutile (vedi docstring di modulo).
    """
    token_id:       int
    request_id:     str
    gating_scores:  list[float]     # score per ogni expert (len = n_experts)
    token_entropy:  float
    bf16_available: bool = False    # deciso dal chiamante (Tier Manager/EAT);
                                     # GCSGGuard non lo deduce da solo
    timestamp:      float = field(default_factory=time.monotonic)


@dataclass
class ShadowExecutionResult:
    """Risultato di una shadow execution."""
    activated:          bool           # shadow execution eseguita?
    shadow_expert_id:   int | None  # expert INT4 usato
    contamination_flag: bool           # questo token è "shadow-contaminated"?
    latency_ms:         float          # overhead shadow execution
    reason_skip:        str | None  # perché shadow non attivata (debug)


class GCSGGuard:
    """Gating Confidence Shadow Guard.

    Args:
        theta_gate:          Soglia gating score (default 0.85).
        theta_entropy:       Soglia entropia token (default 0.70).
        theta_contamination: Soglia tasso contaminazione KV-Cache, per
                              request_id (default 0.05).
        shadow_pool_size:    Numero di shadow expert INT4 richiesti (default 2,
                              vedi memory math nel docstring del modulo).
        per_expert_vram_gb:  VRAM stimata per expert INT4 (default 3.0 GB).
        min_headroom_gb:     Margine oltre al budget expert per CUDA context e
                              frammentazione (default 1.0 GB).
        gpu_index:           Indice GPU da controllare (default 0, unica su dev).
        check_vram:          Se True (default), verifica la VRAM disponibile via
                              NVML e adatta shadow_pool_size di conseguenza —
                              abbassato se serve, RuntimeError solo se non entra
                              nemmeno 1 expert. Disattivare solo nei unit test
                              che non toccano una GPU reale.
    """

    def __init__(
        self,
        theta_gate:          float = 0.85,
        theta_entropy:       float = 0.70,
        theta_contamination: float = 0.05,
        shadow_pool_size:    int = 2,
        per_expert_vram_gb:  float = 3.0,
        min_headroom_gb:     float = 1.0,
        gpu_index:           int = 0,
        check_vram:          bool = True,
    ) -> None:
        self.theta_gate          = theta_gate
        self.theta_entropy       = theta_entropy
        self.theta_contamination = theta_contamination
        self.shadow_pool_size    = shadow_pool_size

        self._total_tokens:          int = 0   # globale, per stats()/contamination_rate() aggregato
        self._contamination_counter: int = 0
        self._request_tokens:        dict[str, int] = {}   # request_id -> token valutati
        self._request_contamination: dict[str, int] = {}   # request_id -> token shadow-contaminati

        if check_vram:
            self.shadow_pool_size = self._check_vram_budget(
                shadow_pool_size, per_expert_vram_gb, min_headroom_gb, gpu_index,
            )

    @staticmethod
    def _check_vram_budget(
        requested_pool_size: int,
        per_expert_vram_gb:  float,
        min_headroom_gb:     float,
        gpu_index:           int,
    ) -> int:
        """Adatta shadow_pool_size alla VRAM libera invece di un fail-fast binario.

        Se il modello attivo ha già consumato molta VRAM (es. > ~18 GB su 24,
        vedi memory math nel docstring del modulo), abbassa shadow_pool_size a
        quanti expert entrano davvero — GCSG degradato ma operativo. Fallisce
        fast (RuntimeError) solo se non entra nemmeno 1 expert: l'alternativa
        sarebbe un OOM a runtime sotto carico, molto peggio da diagnosticare.

        Returns:
            shadow_pool_size effettivo (<= requested_pool_size).
        """
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            free_gb = pynvml.nvmlDeviceGetMemoryInfo(handle).free / (1024 ** 3)
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            return requested_pool_size  # nessuna GPU/driver NVML — non bloccare (es. CI)

        usable_experts = int((free_gb - min_headroom_gb) // per_expert_vram_gb)

        if usable_experts < 1:
            raise RuntimeError(
                f"GCSG: {free_gb:.1f} GB VRAM libera su GPU {gpu_index} — non "
                f"basta nemmeno per 1 shadow expert INT4 (~{per_expert_vram_gb:.1f} GB "
                f"+ {min_headroom_gb:.1f} GB headroom richiesti). GCSG non può avviarsi."
            )

        effective = min(requested_pool_size, usable_experts)
        if effective < requested_pool_size:
            log.warning(
                "GCSG: shadow_pool_size abbassato da %d a %d — solo %.1f GB VRAM "
                "libera su GPU %d (modello attivo ha consumato più del previsto).",
                requested_pool_size, effective, free_gb, gpu_index,
            )
        return effective

    # ── core logic ─────────────────────────────────────────────────────────────

    def should_activate_shadow(self, ctx: GatingContext) -> tuple[bool, str]:
        """Decide se attivare shadow execution per un token.

        Ogni chiamata conta come una valutazione (aggiorna il denominatore di
        contamination_rate(ctx.request_id)), indipendentemente dall'esito —
        run_shadow() aggiorna solo il numeratore quando lo shadow path viene
        effettivamente eseguito.

        Returns:
            (should_activate, reason): reason descrive la condizione bloccante
            se should_activate è False.
        """
        self._total_tokens += 1
        self._request_tokens[ctx.request_id] = self._request_tokens.get(ctx.request_id, 0) + 1

        if ctx.bf16_available:
            return False, "bf16_available"

        top_gating_score = max(ctx.gating_scores) if ctx.gating_scores else 0.0
        if top_gating_score <= self.theta_gate:
            return False, f"gating_score {top_gating_score:.3f} <= theta_gate {self.theta_gate}"

        if ctx.token_entropy >= self.theta_entropy:
            return False, f"token_entropy {ctx.token_entropy:.3f} >= theta_entropy {self.theta_entropy}"

        contamination = self.contamination_rate(ctx.request_id)
        if contamination >= self.theta_contamination:
            return False, f"contamination {contamination:.3f} >= theta_contamination {self.theta_contamination}"

        return True, "all conditions met"

    def run_shadow(
        self,
        ctx: GatingContext,
        shadow_pool: dict[int, object],
        hidden_states: Any,
        layer_id: int,
    ) -> ShadowExecutionResult:
        """Esegue shadow execution con expert INT4 dal shadow_pool.

        Sceglie, tra gli expert col gating score più alto, il primo
        effettivamente presente in shadow_pool (il router potrebbe preferire
        un expert non cachato — in quel caso si scende in classifica finché
        non se ne trova uno disponibile, o si desiste).

        hidden_states/layer_id sono deliberatamente FUORI da GatingContext:
        GatingContext è il contesto della decisione (score, entropy,
        contamination), non dell'esecuzione — hidden_states e layer_id
        appartengono al forward pass dello shadow expert, non alla domanda
        "va attivato?". Aggiunti qui (2026-08-09, integrazione reale) senza
        toccare GatingContext/ShadowExecutionResult: zero impatto sui
        chiamanti esistenti di should_activate_shadow o sui campi già testati.

        Args:
            ctx:           GatingContext del token corrente (decisione).
            shadow_pool:   Dict expert_id → callable(hidden_states, layer_id)
                           -> output. Costruito da _load_shadow_pool().
            hidden_states: Tensore di input per il layer corrente (esecuzione).
            layer_id:      Indice del layer corrente — un "expert i" ha pesi
                           diversi per layer (vedi memory math nel docstring
                           di modulo), lo shadow_pool callable dispatcha di
                           conseguenza.

        Returns:
            ShadowExecutionResult con flag contaminazione e latenza.
        """
        start = time.monotonic()
        ranked_experts = sorted(
            range(len(ctx.gating_scores)), key=lambda i: ctx.gating_scores[i], reverse=True,
        )
        shadow_expert_id = next((e for e in ranked_experts if e in shadow_pool), None)

        if shadow_expert_id is None:
            return ShadowExecutionResult(
                activated=False,
                shadow_expert_id=None,
                contamination_flag=False,
                latency_ms=(time.monotonic() - start) * 1000,
                reason_skip="no gated expert present in shadow_pool",
            )

        shadow_pool[shadow_expert_id](hidden_states, layer_id)   # forward INT4 reale

        self._contamination_counter += 1
        self._request_contamination[ctx.request_id] = (
            self._request_contamination.get(ctx.request_id, 0) + 1
        )

        return ShadowExecutionResult(
            activated=True,
            shadow_expert_id=shadow_expert_id,
            contamination_flag=True,
            latency_ms=(time.monotonic() - start) * 1000,
            reason_skip=None,
        )

    # ── KV-Cache contamination tracking ───────────────────────────────────────

    def contamination_rate(self, request_id: str | None = None) -> float:
        """Tasso contaminazione KV-Cache (0.0–1.0).

        Args:
            request_id: Se fornito, tasso per quella richiesta soltanto —
                        questo è il valore che should_activate_shadow()
                        confronta con θ_contamination. Se None, aggregato su
                        tutte le richieste tracciate (solo per stats()/
                        osservabilità, non per decisioni per-token: un
                        aggregato mischierebbe sessioni diverse, esattamente
                        il problema che request_id risolve).
        """
        if request_id is not None:
            tokens = self._request_tokens.get(request_id, 0)
            contaminated = self._request_contamination.get(request_id, 0)
            return contaminated / tokens if tokens else 0.0
        return self._contamination_counter / self._total_tokens if self._total_tokens else 0.0

    def reset_contamination_counter(self, request_id: str | None = None) -> None:
        """Reset al cambio sessione.

        Args:
            request_id: Se fornito, reset solo per quella richiesta (fine
                        sequenza — libera anche la entry nei dict interni,
                        non solo azzera). Se None, reset globale completo.
        """
        if request_id is not None:
            self._request_tokens.pop(request_id, None)
            self._request_contamination.pop(request_id, None)
            return
        self._total_tokens = 0
        self._contamination_counter = 0
        self._request_tokens.clear()
        self._request_contamination.clear()

    # ── calibration ────────────────────────────────────────────────────────────

    def update_thresholds(self, theta_gate: float | None = None,
                          theta_entropy: float | None = None,
                          theta_contamination: float | None = None) -> None:
        """Aggiorna le soglie a runtime (grid search Sprint 3)."""
        if theta_gate is not None:
            self.theta_gate = theta_gate
        if theta_entropy is not None:
            self.theta_entropy = theta_entropy
        if theta_contamination is not None:
            self.theta_contamination = theta_contamination

    # ── stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Metriche: activation rate, contamination rate, latenza shadow."""
        return {
            "total_tokens_evaluated": self._total_tokens,
            "shadow_activations":     self._contamination_counter,
            "activation_rate": (
                self._contamination_counter / self._total_tokens if self._total_tokens else 0.0
            ),
            "contamination_rate_aggregate": self.contamination_rate(),
            "tracked_requests": len(self._request_tokens),
            "shadow_pool_size": self.shadow_pool_size,
            "thresholds": {
                "theta_gate": self.theta_gate,
                "theta_entropy": self.theta_entropy,
                "theta_contamination": self.theta_contamination,
            },
        }


def _quantize_int4(weight: Any) -> tuple[Any, float]:
    """Quantizzazione simmetrica per-tensore a range INT4 [-8,7].

    Salvata come int8 (nessun bit-packing reale) — vedi _load_shadow_pool
    per il perché: dimostra la correttezza numerica del round-trip
    quantizza/dequantizza, non il risparmio di memoria di un kernel INT4
    packed vero.
    """
    import torch

    max_abs = weight.abs().max()
    scale = float(max_abs / 7.0) if max_abs > 0 else 1.0
    quantized = torch.clamp(torch.round(weight / scale), -8, 7).to(torch.int8)
    return quantized, scale


# ── vendorizzato da casper-hansen/AutoAWQ, awq/utils/packing_utils.py ────────
# (MIT license) — issue #33 Fase 6a. Verificato contro il sorgente reale via
# GitHub API il 2026-08-17 (non reimplementato da zero), poi verificato
# NUMERICAMENTE contro il kernel CUDA AWQ reale su un expert del checkpoint
# di produzione — errore relativo L2 ~0.0005 (LOGBOOK_ISSUE33.MD "Passo 2"),
# non solo fidandosi che il codice vendorizzato fosse corretto perché
# proviene da un progetto reale. _AWQ_REVERSE_ORDER è il punto che rischiava
# di più: AWQ impacchetta i nibble in ordine interleaved
# ([0,4,1,5,2,6,3,7]), non sequenziale — un dettaglio facile da sbagliare
# reimplementando da zero, qui riusato as-is.

_AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def _awq_unpack(qweight: Any, qzeros: Any, bits: int) -> tuple[Any, Any]:
    import torch

    shifts = torch.arange(0, 32, bits, device=qzeros.device)
    iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(torch.int8)
    iweights = iweights.view(iweights.shape[0], -1)
    izeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(torch.int8)
    izeros = izeros.view(izeros.shape[0], -1)
    return iweights, izeros


def _awq_reverse_order(iweights: Any, izeros: Any, bits: int) -> tuple[Any, Any]:
    import torch

    reverse_order_tensor = torch.arange(iweights.shape[-1], dtype=torch.int32, device=izeros.device)
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, _AWQ_REVERSE_ORDER]
    reverse_order_tensor = reverse_order_tensor.view(-1)
    izeros = izeros[:, reverse_order_tensor]
    iweights = iweights[:, reverse_order_tensor]
    return iweights, izeros


def _dequantize_awq_gemm(qweight: Any, qzeros: Any, scales: Any, bits: int, group_size: int) -> Any:
    """Dequantizza pesi AWQ formato GEMM. Layout risultato: (in_features,
    out_features) — convenzione AWQ "y = x @ w", OPPOSTA a nn.Linear/
    _ShadowExpertINT4 (out_features, in_features) — chi chiama questa
    funzione deve trasporre esplicitamente (verificato empiricamente
    contro il config reale del checkpoint, non assunto dalla
    documentazione generica — vedi LOGBOOK_ISSUE33.MD "Passo 1")."""
    import torch

    iweight, izeros = _awq_unpack(qweight, qzeros, bits)
    iweight, izeros = _awq_reverse_order(iweight, izeros, bits)
    iweight = torch.bitwise_and(iweight, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)
    scales_e = scales.repeat_interleave(group_size, dim=0)
    izeros_e = izeros.repeat_interleave(group_size, dim=0)
    return (iweight - izeros_e) * scales_e


def _dequantize_awq_linear_to_fp32(linear: Any) -> Any:
    """Dequantizza un singolo layer Linear AWQ-packed (qweight/qzeros/
    scales) a fp32, layout nn.Linear-style (out_features, in_features) —
    .T esplicito + .contiguous() (verificato 4.4x più lento senza,
    LOGBOOK_ISSUE33.MD "consigli esterni vagliati" — non un dettaglio
    trascurabile).

    bits/group_size derivati dalle shape REALI dei tensori
    (qweight/scales), non da un file di config esterno — questa funzione
    non sa e non deve sapere dove vive il checkpoint su disco: la
    quantizzazione AWQ GEMM codifica queste informazioni nelle shape
    stesse. pack_factor = out_features // qweight.shape[1] (colonne
    impacchettate), bits = 32 // pack_factor; group_size =
    qweight.shape[0] // scales.shape[0] (righe per gruppo di scale
    condiviso).
    """
    import torch

    qweight = linear.qweight.detach().cpu()
    qzeros = linear.qzeros.detach().cpu()
    scales = linear.scales.detach().cpu()

    out_features = scales.shape[1]
    pack_factor = out_features // qweight.shape[1]
    bits = 32 // pack_factor
    group_size = qweight.shape[0] // scales.shape[0]

    dequantized = _dequantize_awq_gemm(qweight, qzeros, scales, bits, group_size)
    return dequantized.T.to(torch.float32).contiguous()


class _ShadowExpertINT4:
    """Callable (hidden_states, layer_id) -> output — forward SwiGLU reale
    su pesi INT4 dequantizzati al volo, un layer alla volta.

    Layout pesi verificato su FusedMoE reale (2026-08-09):
        w13_weight[e]: (2*intermediate_size, hidden_size) — concat(gate_proj,
            up_proj) lungo dim 0. Ordine gate-poi-up confermato via
            FusedMoE.weight_loader: shard_id "w1" (gate) e "w3" (up) scrivono
            entrambi su shard_dim=0 di questo stesso parametro, "w1" per
            primo nella convenzione HF/vLLM standard.
        w2_weight[e]: (hidden_size, intermediate_size) — down_proj.
    Entrambi in layout nn.Linear-style (out_features, in_features): il
    forward usa x @ w.T, non x @ w.

    Cast+scale memoizzati per (layer_id, dtype) (issue #33 Fase 6a,
    2026-08-17 — trovato durante il primo run reale cpu-offload sul path
    AWQ, mai completato in 35+ minuti su 16 prompt): `.to(dtype) * scale`
    ricalcolava l'intero tensore w13/w2 ad OGNI chiamata, invece che una
    volta sola. Sul path INT4 originale (Fase 1) questo era quasi gratis
    (sorgente int8, piccola). Sul path fp32-cache di Fase 6a, il sorgente
    è già un tensore fp32 di centinaia di MB per expert-layer — ricastarlo
    e rimoltiplicarlo ad ogni token, per ogni layer, per ogni expert
    freddo, è un costo che scala con token×layer×expert-freddi invece che
    un costo one-time. Numericamente IDENTICO a prima: dato che lo scale
    è sempre esattamente 1.0 su entrambi i path noti (INT4: il valore
    reale è nella quantizzazione, non in uno scale runtime variabile;
    fp32-cache: i pesi sono già in unità reali), castare una volta a
    build-lazy invece che ad ogni call produce lo STESSO tensore fp16
    finale — non un'approssimazione, solo lo stesso calcolo fatto una
    volta anziché N.
    """

    def __init__(
        self,
        per_layer_w13: list[tuple[Any, float]],
        per_layer_w2: list[tuple[Any, float]],
    ) -> None:
        self._per_layer_w13 = per_layer_w13
        self._per_layer_w2 = per_layer_w2
        self._resolved_cache: dict[tuple[int, Any], tuple[Any, Any]] = {}

    def __call__(self, hidden_states: Any, layer_id: int) -> Any:
        import torch.nn.functional as F

        cache_key = (layer_id, hidden_states.dtype)
        resolved = self._resolved_cache.get(cache_key)
        if resolved is None:
            w13_q, w13_scale = self._per_layer_w13[layer_id]
            w2_q, w2_scale = self._per_layer_w2[layer_id]
            w13 = w13_q.to(hidden_states.dtype) * w13_scale
            w2 = w2_q.to(hidden_states.dtype) * w2_scale
            resolved = (w13, w2)
            self._resolved_cache[cache_key] = resolved
        w13, w2 = resolved

        intermediate_size = w2.shape[-1]
        gate_up = hidden_states @ w13.T
        gate, up = gate_up.split(intermediate_size, dim=-1)
        activated = F.silu(gate) * up
        return activated @ w2.T


class _AWQShadowExpert:
    """Callable (hidden_states, layer_id) -> output — delega al modulo
    MixtralMLP quantizzato reale, un layer alla volta.

    Usato quando block_sparse_moe.experts è una ModuleList di MixtralMLP
    (checkpoint AWQ pre-quantizzato, vllm.model_executor.models.
    mixtral_quant — verificato 2026-08-09 su Mixtral-8x7B-Instruct-v0.1-AWQ
    reale: w1/w2/w3 con qweight/qzeros/scales AWQ packed, non FusedMoE/
    w13_weight come sul modello tiny non quantizzato). Zero weight
    extraction, zero dequant manuale: il forward di MixtralMLP gestisce già
    la dequantizzazione coi kernel AWQ interni di vLLM, che funzionano. Lo
    "shadow" qui è un secondo forward attraverso lo stesso expert già
    caricato (nessuna copia separata in INT4 simulato come _ShadowExpertINT4)
    — accettabile per la validazione: misura il costo/comportamento
    dell'attivazione shadow, non richiede una replica fisica del peso.
    """

    def __init__(self, modules_per_layer: list[Any]) -> None:
        self._modules = modules_per_layer

    def __call__(self, hidden_states: Any, layer_id: int) -> Any:
        return self._modules[layer_id](hidden_states)


class _PinnedMarlinExperts:
    """Vista GPU-resident di un sottoinsieme di expert estratto da un
    FusedMoE Marlin-packed di UN layer — copia sincrona reale (.to('cuda'),
    non il .to(..., non_blocking=True) di vllm.model_executor.models.utils.
    maybe_offload_to_cpu) dei soli tensori che AWQMoEMethod.apply() legge
    da `layer`. Letto il corpo completo di apply() (vllm==0.6.6.post1)
    prima di scrivere questa classe: la chiamata reale al kernel è

        torch.ops.vllm.fused_marlin_moe(x, layer.w13_qweight,
            layer.w2_qweight, layer.w13_scales, layer.w2_scales,
            router_logits, topk_weights, topk_ids,
            w1_zeros=layer.w13_qzeros, w2_zeros=layer.w2_qzeros,
            num_bits=...)

    — esattamente questi sei tensori, nessun'altra dipendenza a runtime
    (g_idx_sort_indices è usato solo durante il repack a load-time, non qui).
    L'asse expert (dim 0) sopravvive intatto al repack Marlin — verificato
    2026-08-10 su casperhansen/mixtral-instruct-awq reale: shape
    (num_experts, ...) su tutti e sei i tensori, prima e dopo
    ops.awq_marlin_moe_repack(). Lo slicing è quindi un taglio pulito lungo
    un asse noto e non tocca la permutazione interna del kernel (che vive
    dentro ogni fetta per-expert, non attraverso l'asse expert) — a
    differenza di invertire il repack stesso (direzione (a1), scartata,
    vedi issue #10), qui non serve interpretare il formato packed.

    Un solo proxy per layer copre TUTTI gli expert_ids del pool insieme
    (non un proxy per expert_id — raddoppierebbe il costo VRAM per nulla,
    dato che _MarlinFusedShadowExpert isola comunque un solo target per
    chiamata via router_logits). Verificato isolatamente prima
    dell'integrazione: scripts/verify_marlin_pinned_proxy.py confronta
    l'output di un expert letto dal proxy (slice pinnata da un layer
    offloaded) contro lo stesso expert letto dal tensore originale intatto
    di un layer non-offloaded — stesso peso, due sorgenti fisiche diverse.

    ATTENZIONE (2026-08-10, trovato DOPO la prima integrazione, vedi
    LOGBOOK): _build_marlin_shadow_pool() costruisce QUESTO proxy solo per
    le layer effettivamente offloaded — MAI per le 26/32 già GPU-resident,
    che riusano il modulo FusedMoE originale direttamente (zero copie,
    zero allocazioni). Il primo tentativo costruiva un proxy per TUTTE le
    32 layer indiscriminatamente: 192 allocazioni GPU piccole invece di 36,
    frammentazione dell'allocatore CUDA sopra un modello già quasi al
    limite dei 24GB, e il profiling di vLLM (determine_num_available_blocks,
    gira DOPO _load_shadow_pool()) si è bloccato cercando memoria
    contigua — container ucciso manualmente dopo 9 minuti a GPU 100%/VRAM
    266MB liberi, senza alcun avanzamento nel log. Non un'ipotesi:
    osservato direttamente (nvidia-smi, log fermo). Questa classe resta
    corretta di per sé (verificata numericamente) — il bug era nel chiamarla
    per layer che non ne avevano bisogno.

    Wiring TierManager (2026-08-12, issue #17): il `.to(device)` diretto
    sopra descritto resta il default (`tensor_promoter=None`) — questa
    classe non è stata riscritta, solo estesa con un hook opzionale
    (vedi `__init__`) che GCSGWorker._build_marlin_shadow_pool() usa
    quando `self._tier_manager` è wired. Deliberatamente conservativo
    proprio per la storia di fragilità sopra: il path di default
    (nessun tier_manager) è BYTE PER BYTE identico a prima di questa
    estensione.
    """

    def __init__(
        self, source_fused: Any, expert_ids: list[int],
        tensor_promoter: Callable[[str, Any], Any] | None = None,
    ) -> None:
        """tensor_promoter (2026-08-12, issue #17): se fornito, sostituisce
        il `.to(device)` diretto per instradare il transfer attraverso
        TierManager quando il GCSGWorker chiamante ha `_tier_manager`
        wired — vedi GCSGWorker._build_marlin_tensor_promoter(). Firma
        `(tensor_name, sliced_cpu_tensor) -> tensore GPU`. None
        (default): comportamento invariato, `.to(device)` diretto come
        prima di questa integrazione — zero rischio per il path già
        validato dal report GCSG."""
        import torch

        device = torch.device("cuda")

        self.quant_method = source_fused.quant_method
        self.top_k = source_fused.top_k
        self.renormalize = source_fused.renormalize
        self.use_grouped_topk = source_fused.use_grouped_topk
        self.topk_group = source_fused.topk_group
        self.num_expert_group = source_fused.num_expert_group
        self.custom_routing_function = source_fused.custom_routing_function
        self.scoring_func = source_fused.scoring_func
        self.e_score_correction_bias = source_fused.e_score_correction_bias

        for name in ("w13_qweight", "w2_qweight", "w13_scales",
                     "w2_scales", "w13_qzeros", "w2_qzeros"):
            source_tensor = getattr(source_fused, name).data
            sliced = source_tensor[expert_ids]
            self.__dict__[name] = (
                tensor_promoter(name, sliced) if tensor_promoter is not None
                else sliced.to(device)
            )


class _MarlinFusedShadowExpert:
    """Callable (hidden_states, layer_id) -> output — isola un singolo
    expert dentro un FusedMoE con pesi Marlin-packed, senza dequantizzare
    nulla a mano.

    Terzo caso, distinto sia da _ShadowExpertINT4 (FusedMoE con pesi fp16
    grezzi w13_weight) sia da _AWQShadowExpert (ModuleList di MixtralMLP):
    qui block_sparse_moe.experts È un FusedMoE (ha num_experts), ma i pesi
    sono Marlin-packed (w13_qweight/w13_scales/w13_qzeros — verificato
    2026-08-09 su casperhansen/mixtral-instruct-awq reale con
    quantization="awq_marlin", vedi GitHub issue #10). A differenza del
    caso ModuleList, qui non esiste un nn.Module separato per ogni
    expert — FusedMoE tiene tutti gli expert in un unico tensore batched e
    il suo forward() fa sempre routing reale su TUTTI gli expert col
    top_k del modello (2 per Mixtral): non c'è "un expert" da chiamare
    direttamente.

    ATTENZIONE (2026-08-09, issue #10): il primo approccio tentato — forzare
    top_k=1 nella chiamata a quant_method.apply() per isolare "davvero" un
    solo expert — provoca un `CUDA error: illegal memory access` dentro il
    kernel Marlin (torch.ops._moe_C.marlin_gemm_moe), riprodotto sia dalla
    chiamata diretta sia da una chiamata di riferimento indipendente
    (vero FusedMoE.forward() con self.top_k monkey-patchato a 1 — stesso
    crash, quindi non un bug di questa classe specifica). Letto il vero
    fused_marlin_moe.py installato (vllm==0.6.6.post1) prima di concludere
    qualunque cosa: i buffer (intermediate_cache2, workspace) sono
    dimensionati dinamicamente per chiamata dai valori reali passati, non
    pre-allocati altrove per il top_k "reale" del modello — quindi la causa
    esatta resta nel kernel CUDA compilato, non ispezionabile da Python, non
    in un mismatch a livello Python verificabile qui. Pattern comunque
    reale e noto: vllm-project/vllm#35922, #32834, #26558 sono tutte
    "illegal memory access" dentro fused_marlin_moe su modelli/config
    diversi (verificate esistenti, non prese per buone a scatola chiusa).

    Fix adottato: NON tocca top_k (resta quello reale del modello, 2 per
    Mixtral — la superficie che ha fatto crashare non viene più toccata).
    Isolamento per via del router_logits: logit fortemente dominante
    sull'expert_id target, logit fortemente negativo su tutti gli altri.
    Con top_k reale invariato, la selezione top-k prende comunque il
    target più un secondo expert arbitrario (qualunque, dato che sono tutti
    ugualmente sfavoriti) — dopo softmax+renormalize il peso del secondo è
    numericamente nullo (~exp(-60), sotto la precisione fp16) e il kernel
    Marlin reale fa dequant+matmul con un contributo del target
    indistinguibile da 1.0. Stesso principio di _AWQShadowExpert (delega al
    codice reale, zero dequantizzazione manuale), ma senza toccare
    l'invariante (top_k) che si è rivelata rischiosa.

    RI-ABILITATA (2026-08-10, issue #10/#16): il crash sopra non era del
    kernel Marlin — era vllm.model_executor.models.utils.
    maybe_offload_to_cpu() (async H2D copy non sicura sotto WSL2 senza
    pin_memory), innescato perché questa classe chiama quant_method.apply()
    su un `fused` i cui tensori potevano essere CPU-resident (layer
    offloaded, cpu_offload_gb=4). _load_shadow_pool() ora garantisce che
    ogni `fused` passato qui sia GPU-resident — un _PinnedMarlinExperts per
    le layer offloaded, il modulo FusedMoE originale (già su GPU, zero
    copie) per le altre — vedi GCSGWorker._build_marlin_shadow_pool().

    Entry PER-LAYER, non un (expert_id, num_experts) uniforme su tutte le
    layer (2026-08-10, secondo giro dopo un hang — vedi ATTENZIONE nella
    docstring di _PinnedMarlinExperts): le layer offloaded usano il proxy a
    len(expert_ids) slot con l'indice LOCALE del target; le layer
    non-offloaded usano il modulo originale a num_experts=8 con l'ID
    GLOBALE — due "forme" diverse per lo stesso expert_id a seconda della
    layer, quindi (fused, expert_id, num_experts) devono viaggiare insieme
    per ogni layer_id, non essere fissati una volta sola per l'istanza.
    """

    _LOGIT_HIGH = 30.0
    _LOGIT_LOW = -30.0

    def __init__(self, layer_entries: list[tuple[Any, int, int]]) -> None:
        """layer_entries[layer_id] = (fused, expert_id_in_fused, num_experts_in_fused)."""
        self._layer_entries = layer_entries

    def __call__(self, hidden_states: Any, layer_id: int) -> Any:
        import torch

        fused, expert_id, num_experts = self._layer_entries[layer_id]
        router_logits = torch.full(
            (hidden_states.shape[0], num_experts),
            self._LOGIT_LOW,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        router_logits[:, expert_id] = self._LOGIT_HIGH
        return fused.quant_method.apply(
            layer=fused,
            x=hidden_states,
            router_logits=router_logits,
            top_k=fused.top_k,
            renormalize=fused.renormalize,
            use_grouped_topk=fused.use_grouped_topk,
            topk_group=fused.topk_group,
            num_expert_group=fused.num_expert_group,
            custom_routing_function=fused.custom_routing_function,
            scoring_func=fused.scoring_func,
            e_score_correction_bias=fused.e_score_correction_bias,
        )


class _RoutedShadowPool:
    """Issue #33 Fase 3 — adapter dict-like passato a GCSGGuard.run_shadow()
    al posto di un dict semplice, per decidere per-expert se instradare al
    pool GPU o al pool CPU/DDR4-resident (issue #33 Fase 2) — senza toccare
    run_shadow() stesso, che non sa nulla di GPU/CPU: vede solo un oggetto
    che risponde a `expert_id in pool` e `pool[expert_id](hidden_states,
    layer_id)`, esattamente il contratto che già rispettava con un dict
    semplice. Stesso principio "zero rischio per il path già validato" di
    ogni altra estensione in questo file.

    La decisione vera vive in GCSGWorker.route_forward() — vedi il suo
    docstring per la logica hot/cold e i fallback.
    """

    def __init__(self, worker: GCSGWorker) -> None:
        self._worker = worker

    def __contains__(self, expert_id: int) -> bool:
        # getattr difensivo su _cpu_shadow_pool — non self._worker._cpu_shadow_pool
        # diretto: un worker di test costruito via __new__() prima
        # dell'integrazione Fase 2/3 (es. TestGCSG._make_worker(), che non
        # assegna _base né _cpu_shadow_pool) cadrebbe altrimenti in
        # __getattr__, che richiede _base — stesso motivo già documentato
        # altrove in questo file per _tier_manager/_n_experts_cached.
        return (
            expert_id in self._worker._shadow_pool
            or expert_id in getattr(self._worker, "_cpu_shadow_pool", {})
        )

    def __getitem__(self, expert_id: int) -> Callable[[Any, int], Any]:
        def _call(hidden_states: Any, layer_id: int) -> Any:
            return self._worker.route_forward(expert_id, layer_id, hidden_states)
        return _call


class GCSGWorker:   # pragma: no cover — richiede vLLM engine live, non unit-testabile
    """Worker vLLM con GCSG cablato.

    Sottoclasse reale di vllm.worker.worker.Worker, istanziata da vLLM stesso
    via EngineArgs(worker_cls="scheduler.gcsg.GCSGWorker") — stringa
    qualname, non la classe (vedi docstring di modulo) — quando si
    costruisce un LLMEngine.
    Segue la sequenza verificata nel docstring di modulo (init_device ->
    load_model -> hook su .gate -> shadow pool).

    Smoke test end-to-end eseguito 2026-08-09 su hf-internal-testing/
    Mixtral-tiny (2 layer, hidden_size=1024, num_local_experts=8 — stessa
    classe MixtralForCausalLM/MixtralMoE del vero Mixtral 8x7B, non un
    modello diverso), NON su Mixtral-8x7B reale: il checkpoint AWQ 4-bit
    reale (~23 GiB) rischia seriamente di non caricare sulla 3090 (24 GiB
    VRAM totale, gpu_memory_utilization=0.9 di default concede 21.6 GiB,
    meno dei soli pesi) — verificarlo su un modello che rischia l'OOM prima
    ancora di finire load_model() non avrebbe isolato bene cosa si sta
    testando. Il tiny model valida la MECCANICA (hook si registrano e
    sparano, request_id reali accessibili, il bookkeeping contamination
    per-request funziona) — NON dice nulla su qualità (MMLU), performance,
    o comportamento VRAM del vero shadow pool. Quei risultati restano
    pending sul modello reale (vedi LOGBOOK 2026-08-09) — dichiararlo così è
    più difendibile in review che un risultato su full model gonfiato da
    cpu_offload_gb non rappresentativo.

    _load_shadow_pool() e il wiring router_logits -> GatingContext (Sprint 3,
    2026-08-09, sessione successiva allo smoke test iniziale) sono ora
    implementati per davvero — estrazione pesi w13/w2 verificata su
    FusedMoE reale, quantizzazione INT4 simmetrica (int8, non packed —
    vedi _quantize_int4), forward SwiGLU reale in _ShadowExpertINT4, e
    should_activate_shadow()/run_shadow() chiamati per ogni riga/token con
    gating_scores e hidden_states REALI dagli hook .gate, non più
    placeholder sintetici in execute_model(). Riverificato con un secondo
    smoke test end-to-end (stesso hf-internal-testing/Mixtral-tiny): GREEN,
    stessa aritmetica di prima (56 token valutati), shadow pool caricato e
    forward SwiGLU diretto verificato numericamente (output finito, shape
    corretta).

    _load_shadow_pool() gestisce TRE strutture, non solo FusedMoE —
    scoperto caricando per la prima volta il vero checkpoint Mixtral-8x7B-
    Instruct-AWQ (2026-08-09): block_sparse_moe.experts è FusedMoE con pesi
    fp16 grezzi (w13_weight) solo su modelli non pre-quantizzati come il
    tiny model; con quantization="awq" semplice è una ModuleList di
    MixtralMLP con pesi GIÀ quantizzati (qweight/qzeros/scales packed, un
    modulo per expert — _AWQShadowExpert); con quantization="awq_marlin" è
    di nuovo un FusedMoE, ma con pesi Marlin-packed
    (w13_qweight/w13_scales/w13_qzeros, non w13_weight — GitHub issue #10).
    _MarlinFusedShadowExpert esiste e delega a AWQMoEMethod.apply() reale,
    ma è disattivato con uno stopgap (2026-08-09): chiamarlo crasha il
    kernel CUDA Marlin, quindi _load_shadow_pool() non lo registra più nello
    shadow_pool per questo path — degrada a hook-only finché issue #10 non
    ha un fix verificato. Gli altri due path delegano al codice reale di
    vLLM dove possibile — nessun bisogno di reimplementare Marlin/AWQ.

    NaN bug (2026-08-09, RISOLTO): il caricamento del checkpoint
    TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ completava senza errori ma
    generate() produceva output degenere (token_id sempre 0, logit NaN) su
    ogni combinazione di quantizzazione/cpu_offload_gb/backend attention
    testata. Causa: checkpoint difettoso, non lo stack OSX-PoC — confermato
    (non solo corroborato) verificando che lo stesso identico stack genera
    testo pulito su casperhansen/mixtral-instruct-awq, e indipendentemente
    contro vllm-project/vllm#2359 (stesso sintomo, stesso file, dal
    2024-01-05, mai risolto per questa quantizzazione specifica). Storia
    completa della diagnosi (isolamento a fette: GCSGWorker, pin_memory/
    WSL, kernel Marlin, cpu_offload_gb) in LOGBOOK 2026-08-09.

    L'import di vllm è locale ai metodi (non al modulo) apposta: gcsg.py deve
    restare importabile — e GCSGGuard testabile — anche in ambienti senza
    vLLM installato (es. CI cpu-tests, che non installa requirements-vllm.txt).

    M1/M2 wiring (2026-08-12, issue #17) — opt-in, default None
    (comportamento invariato, byte per byte identico a prima di questa
    integrazione — zero rischio per il path Marlin già validato dal
    report GCSG, 72.28%/72.3% su Linux reale, Sprint 4).

    Attivazione: quando vLLM costruisce questo worker da sé (il caso
    reale — worker_cls="scheduler.gcsg.GCSGWorker" risolto internamente,
    vedi punto 1 sopra), non c'è modo per il chiamante di passare un
    kwarg extra al costruttore — usare
    GCSGWorker.configure_tier_manager(tier_manager) PRIMA di costruire
    LLM(...)/EngineArgs(...). Passare tier_manager= direttamente al
    costruttore resta valido per chi costruisce GCSGWorker senza passare
    da vLLM (es. i test). Con un TierManager attivo (in un modo o
    nell'altro):

        - EAT viene seeded a load_model() con una entry (expert_id,
          layer_id) per ogni combinazione reale del modello, a Tier.DDR4
          (_seed_eat_entries()).
        - La selezione di QUALI expert entrano nello shadow pool diventa
          EAT-driven (hotness aggregata per expert_id) invece del
          round-robin placeholder — _select_shadow_expert_ids(). Al primo
          load, senza traffico reale ancora accumulato, degrada
          onestamente a un ordine equivalente al round-robin (proprietà
          di un cold start, non un difetto nascosto).
        - Ogni token instradato alimenta EAT con traffico reale
          (EAT.access() sul top-1 expert per (token, layer),
          indipendentemente da should_activate_shadow) —
          _evaluate_gcsg_for_rows(). Questa è la "traffico concorrente
          reale" di cui M1 (issue #1/#2/#4) ha bisogno per essere
          misurato, non più solo unit test sintetici.
        - Il transfer GPU sia del path AWQ ModuleList (path 3,
          _promote_module_via_tier_manager()) sia del path Marlin (path
          2, quello effettivamente usato dal checkpoint reale del report
          GCSG — _build_marlin_tensor_promoter()) passa da
          TierManager.promote_live_tensor() invece di un .to(device)
          diretto. Il path Marlin è stato wired più tardi, deliberatamente
          dopo che il path AWQ era già stato verificato due volte su
          hardware reale con determinismo perfetto — vedi la nota nel
          punto di chiamata in _load_shadow_pool() e la docstring di
          _build_marlin_shadow_pool() per la sequenza.
        - refresh_shadow_pool_selection() ricalcola/ricarica il pool da
          hotness aggiornata, ma non è agganciata a nessun trigger
          automatico — richiede prima il profiling di
          promote()/evict() su hardware reale (Sprint 4 sotto-obiettivo 4).

    Issue #33 Fase 2 — pool CPU-resident (2026-08-17): quando un
    TierManager è wired, il path 1 (FusedMoE fp16 grezzo) è in uso, E
    self._cpu_offload_enabled è True (Fase 4, default False —
    configure_cpu_offload()/enable_cpu_offload=, vedi il commento su
    _pending_enable_cpu_offload), _load_shadow_pool() costruisce ANCHE
    self._cpu_shadow_pool — _build_cpu_shadow_pool() esegue lo stesso
    loop di estrazione/quantizzazione del pool GPU ma senza il
    `.to("cuda")` forzato: i pesi restano DDR4-resident, _ShadowExpertINT4
    li usa senza modifiche (già device-agnostic, Fase 1). Parallelo a
    self._shadow_pool, non un suo sostituto. Il routing vero e proprio
    (QUALE pool usare per un expert presente in entrambi) è Fase 3
    (route_forward()) — Fase 4 ha chiuso il gate su overhead di dispatch
    (trascurabile) e stabilità pin_memory (non pertinente, questo path
    non pinna nulla), ma resta comunque dietro il flag esplicito sopra:
    l'overhead-non-trascurabile e la sicurezza tecnica non implicano da
    soli che la funzionalità debba essere live di default ovunque un
    TierManager esista per qualunque altro motivo (es. solo hotness
    tracking EAT lato GPU, issue #17) — l'impatto reale va ancora
    misurato e documentato (Fase 5) prima di quella decisione. Alla
    chiusura di Fase 4 solo il path 1 era coperto: i path 2/3 (Marlin,
    AWQ ModuleList) delegano a kernel CUDA reali (quant_method.apply())
    mai validati su CPU, fuori scope per quella fase.

    Issue #33 Fase 6a (2026-08-17) chiude il gap per il path 3
    (AWQ ModuleList) — l'UNICO path che il checkpoint reale di produzione
    (casperhansen/mixtral-instruct-awq) usa: path 1/is_fused non si
    applica mai a un checkpoint AWQ pre-quantizzato. _dequantize_awq_gemm()
    (vendorizzata da AutoAWQ, MIT, verificata contro il kernel CUDA reale
    — errore relativo L2 ~0.0005) dequantizza i pesi qweight/qzeros/scales
    a fp32, cache CPU-resident via _build_cpu_shadow_pool_awq() — NON
    INT4: ri-quantizzare per-tensore un formato per-gruppo (AWQ,
    group_size derivato dalle shape reali) distrugge il segnale (errore
    relativo ~0.95, misurato attraverso il forward SwiGLU completo — vedi
    TestQuantizeInt4KnownLimitation in tests/test_cpu_kernel.py). Costo:
    ~21GB RAM per expert su 32 layer, ~1.1-1.2s di dequant one-time per
    expert-layer (misurato sul checkpoint reale). Path Marlin (path 2)
    resta fuori scope: nessun checkpoint reale usato da questo progetto
    lo esercita finora.

    Stato di verifica, dichiarato esplicitamente per lo stesso motivo di
    ogni altra claim in questo file: la logica pura Python (selezione,
    seeding, aggregazione hotness) è la stessa testabile via CPU unit test
    di sempre. Il bridging asyncio.run() dentro load_model() e il transfer
    GPU reale via TierManager NON sono stati eseguiti su hardware reale in
    questa sessione (nessuna GPU disponibile qui) — scritti secondo la
    stessa logica già verificata per .to('cuda')/GPUTransfer.to_vram(),
    ma da confermare end-to-end sul pod prima di fidarsene per un run
    MMLU comparabile ai precedenti.
    """

    # Tetto per captured_router_logits (osservabilità smoke-test) — vedi il
    # commento su self.captured_router_logits in __init__ per la cronologia
    # del bug di crescita illimitata trovato 2026-08-10.
    _MAX_CAPTURED_ROUTER_LOGITS = 1000

    # Configurazione "pending" per tier_manager (2026-08-12, issue #17).
    #
    # PROBLEMA REALE, non ipotetico: vLLM costruisce GCSGWorker da solo,
    # risolvendo worker_cls come stringa qualname (vedi docstring di
    # modulo, punto 1) e passandogli i SUOI argomenti standard — non c'è
    # alcun punto in EngineArgs/LLM() dove un chiamante possa iniettare un
    # kwarg extra come tier_manager= nel costruttore. Verificato
    # negativamente: NESSUNO degli script esistenti in scripts/ che usano
    # worker_cls="scheduler.gcsg.GCSGWorker" (eval_mmlu_gcsg.py,
    # probe_kv_blocks.py, smoke_test_gcsg_worker.py,
    # smoke_test_gcsg_mixtral8x7b.py, verify_shadow_pool_pinning_e2e.py)
    # passa mai un argomento extra attraverso quel path — se ne esistesse
    # uno, ci si aspetterebbe almeno un precedente.
    #
    # Fix: configure_tier_manager() imposta un valore a livello di classe
    # PRIMA di costruire LLM(...)/EngineArgs(...); __init__ lo usa come
    # fallback quando tier_manager= non è stato passato esplicitamente
    # (che resta il modo diretto per chi costruisce GCSGWorker da sé, es.
    # i test — vedi tests/test_scheduler.py::TestGCSGTierManagerWiring).
    _pending_tier_manager: TierManager | None = None

    @classmethod
    def configure_tier_manager(cls, tier_manager: TierManager | None) -> None:
        """Imposta il TierManager che il PROSSIMO GCSGWorker costruito da
        vLLM userà. Va chiamato PRIMA di LLM(...)/EngineArgs(...) — vedi
        il commento sopra _pending_tier_manager per perché serve.

        Stato globale a livello di classe, non di istanza: onesto sui
        suoi limiti, non nascosto — un solo processo costruisce un solo
        LLM/worker alla volta in tutti gli usi reali di questo progetto
        (uno script = un worker), quindi non c'è oggi un caso d'uso reale
        per più TierManager pendenti in parallelo nello stesso processo.
        Se dovesse servire, il fix è passare tier_manager= direttamente
        (bypassando questo meccanismo) a chi costruisce GCSGWorker senza
        passare da vLLM.
        """
        cls._pending_tier_manager = tier_manager

    # Configurazione "pending" per il pool CPU/DDR4-resident (issue #33
    # Fase 2/3/4, 2026-08-17) — stesso meccanismo di _pending_tier_manager
    # sopra, stesso motivo (vLLM costruisce GCSGWorker da solo).
    #
    # Default False DELIBERATAMENTE, non solo "se tier_manager è wired
    # allora attiva anche il pool CPU": Fase 4 ha chiuso il gate
    # sull'overhead di dispatch (trascurabile, ~0.08% del tempo di
    # compute — vedi benchmarks/bench_route_forward.py) e sul rischio di
    # pin_memory (non pertinente, questo path non pinna nulla), ma questo
    # NON significa che il routing CPU/DDR4-resident debba attivarsi
    # automaticamente ogni volta che qualcuno wira un TierManager per
    # tutt'altro motivo (es. solo per l'hotness tracking EAT lato GPU,
    # issue #17). È una funzionalità distinta, non ancora misurata in
    # produzione (Fase 5 non fatta) — resta spenta finché non viene
    # esplicitamente richiesta, cosicché l'impatto reale (positivo o
    # negativo) possa essere testato e documentato prima di diventare il
    # default, non deciso a priori qui.
    _pending_enable_cpu_offload: bool = False

    @classmethod
    def configure_cpu_offload(cls, enabled: bool) -> None:
        """Abilita/disabilita il routing CPU/DDR4-resident (issue #33) per
        il PROSSIMO GCSGWorker costruito da vLLM. Va chiamato PRIMA di
        LLM(...)/EngineArgs(...), insieme a configure_tier_manager() (un
        TierManager wired è comunque un prerequisito — questo flag da solo
        non basta, vedi _load_shadow_pool()).

        Default False: vedi il commento su _pending_enable_cpu_offload per
        perché non è legato automaticamente alla presenza di un
        TierManager.
        """
        cls._pending_enable_cpu_offload = enabled

    def __init__(
        self, *args, guard: GCSGGuard | None = None,
        tier_manager: TierManager | None = None,
        enable_cpu_offload: bool | None = None, **kwargs,
    ) -> None:
        from vllm.worker.worker import Worker  # import locale, vedi docstring classe
        self._base = Worker(*args, **kwargs)
        self.guard = guard or GCSGGuard()
        # M2/M1 wiring (2026-08-12, issue #17) — opt-in, default None:
        # con tier_manager=None (default) il comportamento è BYTE PER BYTE
        # identico a prima di questa integrazione (round-robin +
        # .to('cuda') diretto) — zero rischio per il path Marlin già
        # validato (report GCSG, 72.28%/72.3% su Linux reale). Vedi
        # _load_shadow_pool()/_select_shadow_expert_ids() per dove diverge
        # quando presente, e la classe docstring per lo stato di verifica.
        # Fallback a _pending_tier_manager (configure_tier_manager()): vedi
        # il commento su quell'attributo per perché serve — vLLM costruisce
        # questo worker da solo, un kwarg esplicito qui non è raggiungibile
        # dall'esterno quando LLM(worker_cls=...) è il chiamante reale.
        self._tier_manager = tier_manager if tier_manager is not None else type(self)._pending_tier_manager
        # Issue #33 Fase 2/3/4 — flag esplicito e SEPARATO da tier_manager:
        # vedi il commento su _pending_enable_cpu_offload per perché il
        # routing CPU/DDR4-resident non si attiva solo perché un
        # TierManager è wired. Default False.
        self._cpu_offload_enabled = (
            enable_cpu_offload if enable_cpu_offload is not None
            else type(self)._pending_enable_cpu_offload
        )
        self._n_experts_cached: int | None = None
        self._shadow_pool: dict[int, object] = {}
        # Issue #33 Fase 2 — pool CPU-resident (DDR4), parallelo a
        # _shadow_pool (GPU), non un suo sostituto: vedi
        # _build_cpu_shadow_pool()/_load_shadow_pool() per come/quando
        # viene popolato (solo con self._tier_manager wired E
        # self._cpu_offload_enabled True — entrambi, non basta uno solo).
        self._cpu_shadow_pool: dict[int, object] = {}
        # Issue #33 Fase 3 — quali expert_id (tra quelli presenti in
        # ENTRAMBI i pool) sono "caldi" in questo momento: vedi
        # _refresh_hot_cold_classification()/route_forward(). Policy
        # dedicata (non tier_manager._policy, privata e potenzialmente
        # LRUPolicy se use_see=False — questa decisione è concettualmente
        # separata dall'eviction VRAM di TierManager, anche se riusa la
        # stessa formula di score via SEEPolicy.classify_hot_cold, Fase 0).
        self._hot_cold_policy = SEEPolicy()
        self._hot_expert_ids: set[int] = set()
        self._gate_hook_handles: list[object] = []

        # Osservabilità smoke-test (2026-08-09) — non usata dal path di
        # produzione, permette di verificare dall'esterno che gli hook
        # sparino davvero e che i request_id reali arrivino a execute_model().
        #
        # BUG REALE trovato 2026-08-10 (issue #10/#16, durante il primo run
        # MMLU con shadow execution davvero attiva): questa lista veniva
        # popolata INCONDIZIONATAMENTE ad ogni hook .gate (ogni layer, ogni
        # forward pass) — un tensore GPU per hit, mai svuotata, tenuto in
        # vita per l'intera durata del processo. Innocuo sugli smoke test
        # (pochi token, tiny model) — mai esercitato su un carico reale
        # finché la shadow execution non ha effettivamente funzionato su un
        # run vero. Sul run MMLU (n=32: 506.784 token valutati) causava
        # crescita illimitata di memoria GPU non più liberabile
        # dall'allocatore di caching di PyTorch — sintomo osservato: run
        # piccoli isolati (n=8/16/32) puliti, run cumulativi nello stesso
        # processo (n=64, o blocchi consecutivi via --chunk-size nello
        # stesso worker) che si bloccano con GPU al 15-25% (pressione di
        # memoria/allocatore, non calcolo saturo) — non un crash immediato,
        # una lenta frammentazione che il chunking da solo non risolve
        # (la lista sopravvive tra le chiamate a generate() nello stesso
        # processo). Cap a _MAX_CAPTURED_ROUTER_LOGITS: nessun consumer
        # (scripts/smoke_test_gcsg_worker.py, scripts/
        # smoke_test_gcsg_mixtral8x7b.py) legge altro che len() e la shape
        # del primo elemento — comportamento identico per gli smoke test
        # (poche decine di hit, mai vicino al tetto), crescita bloccata sui
        # carichi reali.
        self.captured_router_logits: list[object] = []   # torch.Tensor per hit, non tipizzato qui per non importare torch al modulo
        self.seen_request_ids: set = set()

        # riga -> request_id per il batch corrente, popolato da execute_model()
        # e consumato dagli hook .gate (_evaluate_gcsg_for_rows) durante la
        # chiamata nested — vedi execute_model().
        self._current_row_request_ids: list[str] = []

    def __getattr__(self, name):
        # Delega tutto ciò che non sovrascriviamo esplicitamente al Worker
        # reale. Guardia esplicita su "_base" (2026-08-12, trovato scrivendo
        # i test per il wiring TierManager/EAT): __getattr__ scatta solo
        # quando l'attributo normale NON è stato trovato — su un
        # GCSGWorker costruito via __new__() nei test, senza _base mai
        # assegnato, self._base qui sopra ricadrebbe di nuovo in
        # __getattr__('_base'), che tenta di nuovo self._base, all'infinito
        # -> RecursionError invece di un pulito AttributeError. Non solo un
        # problema di test: qualunque accesso ad attributo mancante su un
        # worker incompletamente costruito avrebbe lo stesso destino.
        # __dict__ bypassa __getattr__ per il check stesso.
        if "_base" not in self.__dict__:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute {name!r} "
                f"(né '_base' è stato impostato — worker costruito senza "
                f"passare da __init__?)"
            )
        return getattr(self._base, name)

    def init_device(self) -> None:
        self._base.init_device()

    def load_model(self) -> None:
        """Carica il modello principale, POI gli hook, POI (best-effort) lo shadow pool.

        Ordine del modello vs. shadow pool verificato in
        vllm.executor.gpu_executor.GPUExecutor: init_device() gira prima di
        load_model() e già misura la VRAM libera pre-modello — se lo shadow
        pool venisse caricato in init_device() (o prima di super().load_model()
        qui), il preflight VRAM di GCSGGuard vedrebbe VRAM libera
        artificiosamente alta.

        _load_shadow_pool() è avvolto in un try/except difensivo — un
        fallimento lì (es. VRAM insufficiente per il modello reale) non deve
        impedire l'avvio del worker: hook/request_id/contamination bookkeeping
        non dipendono dallo shadow pool e restano verificabili comunque.
        """
        self._base.load_model()
        self._register_gate_hooks()
        if self._tier_manager is not None:
            try:
                self._seed_eat_entries()
            except Exception as e:
                log.warning(
                    "GCSG: seed EAT fallito (%s) — hotness tracking disattivato "
                    "per questa sessione; con tier_manager wired ma EAT non "
                    "seeded, _select_shadow_expert_ids() degrada comunque a "
                    "round-robin (stesso fallback del caso tier_manager=None).", e,
                )
        try:
            self._load_shadow_pool()
        except Exception as e:
            log.warning(
                "GCSG: shadow pool non caricato (%s) — GCSGWorker gira in "
                "modalità hook-only: hook/request_id/contamination bookkeeping "
                "restano verificabili, nessuna shadow execution possibile.", e,
            )

    def _load_shadow_pool(self) -> None:
        """Costruisce shadow_pool_size shadow expert, da TUTTI i layer del
        modello caricato — TRE percorsi, scelti in base a come vLLM ha
        istanziato gli expert per QUESTO checkpoint:

        1. block_sparse_moe.experts è un FusedMoE con pesi fp16 grezzi
           (w13_weight/w2_weight) — modello non pre-quantizzato (es. il tiny
           model di test). Estrae e quantizza (INT4 simulato, int8 non
           packed — vedi _quantize_int4/_ShadowExpertINT4) i pesi, poi
           esegue SwiGLU manualmente sui pesi shadow dequantizzati.

        2. block_sparse_moe.experts è un FusedMoE con pesi Marlin-packed
           (w13_qweight/w13_scales/w13_qzeros — checkpoint AWQ caricato con
           quantization="awq_marlin", GitHub issue #10). _MarlinFusedShadowExpert
           delega a quant_method.apply() reale con router_logits one-hot —
           vedi la sua docstring per la cronologia del crash e del fix.

        3. block_sparse_moe.experts è una ModuleList di MixtralMLP con pesi
           GIÀ quantizzati (AWQ qweight/qzeros/scales packed, un modulo per
           expert — verificato 2026-08-09 su Mixtral-8x7B-Instruct-v0.1-AWQ
           con quantization="awq" semplice). _AWQShadowExpert delega
           direttamente al modulo reale.

        RI-ABILITATI (2026-08-10, issue #10/#16, dopo lo stopgap 2026-08-09/
        2026-08-10): entrambi i path 2 e 3 crashavano (CUDA illegal memory
        access) non per un bug nel kernel Marlin o in _AWQShadowExpert, ma
        perché chiamare uno shadow expert offloaded su CPU direttamente —
        fuori dal forward sequenziale del modello — bypassa
        vllm.model_executor.models.utils.maybe_offload_to_cpu() (che wrappa
        SOLO il forward dell'intera decoder layer, mai i singoli moduli
        expert/FusedMoE — verificato empiricamente, scripts/
        map_offload_state.py) e finisce per operare su tensori CPU-resident
        con un kernel CUDA. Fix: prima di registrare un expert_id nel pool,
        assicurarsi che sia GPU-resident per costruzione —
        _pin_awq_expert_to_gpu() (path 3, sposta i moduli con una copia
        sincrona reale) o _build_marlin_shadow_pool() (path 2, slice
        GPU-resident via _PinnedMarlinExperts — l'asse expert sopravvive
        intatto al repack Marlin, verificato). Se il pinning fallisce per
        anche un solo layer, l'intero expert_id resta fuori dal pool —
        niente pool "a metà" con alcune layer pinnate e altre no.

        Costo VRAM (shadow_pool_size=2, cpu_offload_gb=4, verificato non
        stimato): con questo checkpoint le prime 6 layer (0-5) sono
        offloaded — layer 0-4 per intero, layer 5 parzialmente (6/8 expert,
        con gli expert 0 e 1 del pool entrambi tra quelli offloaded).
        Pinnare gli expert 0/1 su tutte e 6 le layer costa ≈1.02 GiB
        aggiuntivi, sempre residenti — contro un margine KV-cache di
        ≈1.16-1.24 GiB nella config di validazione. Tema noto, non
        risolto qui: vedi max_num_seqs nei entrypoint di validazione.

        Selezione expert: placeholder round-robin (range(shadow_pool_size))
        quando self._tier_manager è None (default, comportamento invariato).
        Con TierManager wired, selezione reale via EAT hotness — vedi
        _select_shadow_expert_ids() (2026-08-12, issue #17) — questo
        metodo resta responsabile solo dell'estrazione/wiring, non della
        policy di scelta.
        """
        model = self._base.model_runner.model
        layers = model.model.layers
        first_experts = layers[0].block_sparse_moe.experts
        is_fused = hasattr(first_experts, "num_experts")
        is_marlin_packed = is_fused and hasattr(first_experts, "w13_qweight")

        n_experts = first_experts.num_experts if is_fused else len(first_experts)
        self._n_experts_cached = n_experts
        expert_ids = self._select_shadow_expert_ids(n_experts)

        if is_marlin_packed:
            # NOTA (2026-08-12, issue #17): il pinning GPU di questo path
            # è stato lasciato .to(device) diretto per la maggior parte
            # della giornata — il meccanismo di pinning più delicato del
            # file (vedi ATTENZIONE nella docstring di _PinnedMarlinExperts
            # — un hang reale da allocatore CUDA frammentato, 2026-08-10),
            # e il path effettivamente usato dal checkpoint reale del
            # report GCSG (casperhansen/mixtral-instruct-awq). Wired anche
            # questo attraverso TierManager più tardi lo stesso giorno,
            # deciso con l'utente solo DOPO che il path AWQ era stato
            # verificato due volte su hardware reale con determinismo
            # perfetto (411/570 identico byte-per-byte su due run) — vedi
            # _build_marlin_shadow_pool()/_build_marlin_tensor_promoter()
            # per il come. self._tier_manager is None (default):
            # comportamento invariato, .to(device) diretto come sempre.
            marlin_pool = self._build_marlin_shadow_pool(layers, expert_ids)
            self._shadow_pool.update(marlin_pool)
            missing = [e for e in expert_ids if e not in marlin_pool]
            if missing:
                log.warning(
                    "GCSG: shadow pool NON caricato per %d expert Marlin-packed "
                    "(%s) su %d layer — pinning GPU fallito (vedi warning "
                    "precedente). Hook-only per questi expert.",
                    len(missing), missing, len(layers),
                )
            if marlin_pool:
                log.info(
                    "GCSG: shadow pool caricato — %d expert Marlin-packed (%s) "
                    "su %d layer, path=FusedMoE-Marlin+pinning GPU esplicito "
                    "(issue #10/#16).",
                    len(marlin_pool), sorted(marlin_pool), len(layers),
                )
            return

        for expert_id in expert_ids:
            if is_fused:
                per_layer_w13 = []
                per_layer_w2 = []
                for layer in layers:
                    experts_module = layer.block_sparse_moe.experts
                    w13 = experts_module.w13_weight.data[expert_id]   # (2*intermediate, hidden)
                    w2 = experts_module.w2_weight.data[expert_id]     # (hidden, intermediate)
                    # Bug reale trovato 2026-08-12 (issue #17, sub-goal 6, prima
                    # esecuzione mai fatta sotto vero offload): a differenza dei
                    # path 2/3, questo loop non pinnava mai esplicitamente in GPU
                    # — path 1 era finora sempre stato verificato solo sul modello
                    # tiny non offloaded, dove w13/w2 erano già CUDA-resident per
                    # costruzione. Sotto cpu_offload_gb reale queste slice restano
                    # CPU-resident se la layer è offloaded, _quantize_int4() le
                    # quantizza sul posto (int8 su CPU) e _ShadowExpertINT4 crasha
                    # al primo generate() reale: "Expected all tensors to be on
                    # the same device, cuda:0 and cpu" nel matmul hidden_states @
                    # w13.T. Stesso pattern di device-check-poi-.to('cuda') già
                    # usato da _pin_awq_expert_to_gpu()/_build_marlin_shadow_pool()
                    # per i path 2/3 quando self._tier_manager è None.
                    if w13.device.type != "cuda":
                        w13 = w13.to("cuda")
                    if w2.device.type != "cuda":
                        w2 = w2.to("cuda")
                    per_layer_w13.append(_quantize_int4(w13))
                    per_layer_w2.append(_quantize_int4(w2))
                self._shadow_pool[expert_id] = _ShadowExpertINT4(per_layer_w13, per_layer_w2)
            else:
                modules_per_layer = self._pin_awq_expert_to_gpu(layers, expert_id)
                if modules_per_layer is not None:
                    self._shadow_pool[expert_id] = _AWQShadowExpert(modules_per_layer)
                # else: expert_id resta fuori dal pool, hook-only per lui —
                # _pin_awq_expert_to_gpu() ha già loggato il motivo.

        if is_fused:
            log.info(
                "GCSG: shadow pool caricato — %d expert (%s) su %d layer, "
                "path=FusedMoE+INT4-simulato.",
                len(expert_ids), expert_ids, len(layers),
            )
            # Issue #33 Fase 2: pool CPU-resident parallelo, stessi
            # expert_ids, stessa classe _ShadowExpertINT4 (device-agnostic,
            # Fase 1) — mai promosso a CUDA. Richiede ENTRAMBI: un
            # TierManager wired (è lavoro di compute-offload DDR4, non ha
            # senso senza un tier system a monte) E il flag esplicito
            # self._cpu_offload_enabled (Fase 4, default False — vedi il
            # commento su _pending_enable_cpu_offload nel costruttore per
            # perché i due non sono la stessa cosa). Vedi
            # _build_cpu_shadow_pool() per il perché niente .to('cuda') qui.
            if (
                getattr(self, "_tier_manager", None) is not None
                and getattr(self, "_cpu_offload_enabled", False)
            ):
                self._cpu_shadow_pool.update(
                    self._build_cpu_shadow_pool(layers, expert_ids),
                )
                log.info(
                    "GCSG: CPU shadow pool (DDR4-resident, issue #33) "
                    "caricato — %d expert (%s) su %d layer.",
                    len(self._cpu_shadow_pool), sorted(self._cpu_shadow_pool),
                    len(layers),
                )
                # Issue #33 Fase 3: ricalcola subito la classificazione
                # hot/cold per i due pool appena costruiti — vedi il
                # docstring del metodo per il perché non ad ogni
                # route_forward().
                self._refresh_hot_cold_classification()
        else:
            missing = [e for e in expert_ids if e not in self._shadow_pool]
            if missing:
                log.warning(
                    "GCSG: shadow pool NON caricato per %d expert AWQ-pre-"
                    "quantizzati (%s) su %d layer — pinning GPU fallito. "
                    "Hook-only per questi expert.",
                    len(missing), missing, len(layers),
                )
            loaded = [e for e in expert_ids if e in self._shadow_pool]
            if loaded:
                log.info(
                    "GCSG: shadow pool caricato — %d expert AWQ-pre-quantizzati "
                    "(%s) su %d layer, path=AWQ-ModuleList+pinning GPU esplicito "
                    "(issue #16).",
                    len(loaded), loaded, len(layers),
                )
            # Issue #33 Fase 6a: mirror CPU-resident per il path REALE del
            # checkpoint di produzione (path 2/3 sono gli unici usati da
            # casperhansen/mixtral-instruct-awq — path 1/is_fused sopra
            # non si applica mai a un checkpoint AWQ pre-quantizzato, solo
            # al modello tiny non quantizzato usato nei test Fase 1/2/3).
            # Costruito per l'INTERO expert_ids selezionato, non solo
            # `loaded`: se il pinning GPU è fallito per un expert (sopra),
            # questo pool CPU gli dà comunque una residenza funzionante
            # invece di lasciarlo hook-only — un miglioramento rispetto al
            # comportamento pre-Fase-6a, non solo un mirror. Stesso gate di
            # is_fused sopra: richiede ENTRAMBI tier_manager wired E il
            # flag esplicito _cpu_offload_enabled (Fase 4, default False).
            # Vedi _build_cpu_shadow_pool_awq() per il perché fp32 (non
            # INT4) e per la derivazione di bits/group_size dalle shape.
            if (
                getattr(self, "_tier_manager", None) is not None
                and getattr(self, "_cpu_offload_enabled", False)
            ):
                self._cpu_shadow_pool.update(
                    self._build_cpu_shadow_pool_awq(layers, expert_ids),
                )
                log.info(
                    "GCSG: CPU shadow pool (DDR4-resident, issue #33 Fase 6a, "
                    "path AWQ-ModuleList) caricato — %d expert (%s) su %d "
                    "layer.",
                    len(self._cpu_shadow_pool), sorted(self._cpu_shadow_pool),
                    len(layers),
                )
                self._refresh_hot_cold_classification()

    def _build_cpu_shadow_pool(
        self, layers: list[Any], expert_ids: list[int],
    ) -> dict[int, object]:
        """Issue #33 Fase 2 — mirror CPU-resident del path 1 (FusedMoE fp16
        grezzo), parallelo a self._shadow_pool (GPU), non un suo sostituto.

        Stesso loop di estrazione/quantizzazione della sezione `is_fused` in
        _load_shadow_pool(), ma SENZA il `.to("cuda")` forzato (bug fix
        2026-08-12, vedi commento lì) — i pesi restano dove sono. Nessuna
        classe nuova: _ShadowExpertINT4.__call__() è già device-agnostic
        (Fase 1, confermato su Xeon 6244 reale, non solo in sandbox — vedi
        LOGBOOK_ISSUE33.MD), un forward su pesi CPU-resident gira su CPU
        senza modifiche al kernel.

        Non forza `.cpu()` su un peso che risultasse già CUDA-resident
        (modello non sotto cpu_offload_gb — caso raro nell'uso reale target
        di questo lavoro): questo metodo non chiama MAI `.to()`, in nessuna
        direzione — sceglie solo di non fare la promozione che fa il path
        GPU sopra. Un D2H esplicito per liberare VRAM sarebbe una decisione
        di eviction, fuori scope per Fase 2 (il routing hot/cold è Fase 3 —
        _select_shadow_expert_ids() decide solo QUALI expert_id entrano
        qui, non la loro residenza fisica).

        Chiamato solo quando self._tier_manager è wired (vedi
        _load_shadow_pool) — opt-in, stesso principio del resto del wiring
        issue #17: comportamento invariato quando tier_manager è None.
        """
        cpu_pool: dict[int, object] = {}
        for expert_id in expert_ids:
            per_layer_w13 = []
            per_layer_w2 = []
            for layer in layers:
                experts_module = layer.block_sparse_moe.experts
                w13 = experts_module.w13_weight.data[expert_id]
                w2 = experts_module.w2_weight.data[expert_id]
                per_layer_w13.append(_quantize_int4(w13))
                per_layer_w2.append(_quantize_int4(w2))
            cpu_pool[expert_id] = _ShadowExpertINT4(per_layer_w13, per_layer_w2)
        return cpu_pool

    def _build_cpu_shadow_pool_awq(
        self, layers: list[Any], expert_ids: list[int],
    ) -> dict[int, object]:
        """Issue #33 Fase 6a — mirror CPU-resident del path AWQ-ModuleList
        (path 2/3, il SOLO path che il checkpoint reale di produzione usa:
        casperhansen/mixtral-instruct-awq è pre-quantizzato AWQ, non c'è
        mai un path fp16 grezzo su hardware reale — path 1/is_fused resta
        rilevante solo per i test sul modello tiny). Prima di oggi tutto
        il lavoro Fase 6a (dequant AWQ, parità numerica contro il kernel
        CUDA reale, pipeline completa) viveva in script standalone,
        MAI collegato a GCSGWorker — questo metodo è il collegamento.

        Cache FP32, non INT4: _quantize_int4() è per-tensore, ma AWQ
        quantizza per-gruppo (group_size derivato dalle shape, tipicamente
        128) — ri-quantizzare pesi già dequantizzati da un formato
        per-gruppo con una griglia per-tensore distrugge il segnale
        (errore relativo L2 misurato ~0.95 attraverso il forward SwiGLU
        completo, non solo sui pesi grezzi — vedi
        TestQuantizeInt4KnownLimitation in tests/test_cpu_kernel.py).
        La cache fp32 invece misura ~0.0005 di errore relativo contro il
        kernel CUDA reale, a costo di più memoria (~21GB per expert su 32
        layer, misurato) — scelta deliberata, non un compromesso
        provvisorio: vedi LOGBOOK_ISSUE33.MD "misura completa della
        pipeline". _ShadowExpertINT4 riusata con scale=1.0 per ogni layer
        (i pesi sono già in unità reali, il moltiplicatore diventa un
        no-op) — nessuna classe nuova, stesso principio di
        _build_cpu_shadow_pool() per il path 1.

        w1/w3 concatenati su dim 0 per formare w13 (stesso layout
        (2*intermediate, hidden) del path 1/FusedMoE) così
        _ShadowExpertINT4.__call__() — che fa lo split silu(w1)*w3 su
        w13 — funziona identico sui due path senza bisogno di una classe
        _AWQShadowExpertCPU dedicata.

        Nessun try/except per-expert: un fallimento di dequant qui è un
        bug reale (shape/attributi inattesi), non uno scenario atteso —
        propaga fino al try/except di alto livello già presente attorno
        a _load_shadow_pool() in load_model(), che degrada a hook-only e
        logga un warning (stesso comportamento di qualunque altra
        eccezione in questo metodo, invariato da prima di Fase 6a).
        """
        import torch

        cpu_pool: dict[int, object] = {}
        for expert_id in expert_ids:
            per_layer_w13 = []
            per_layer_w2 = []
            for layer in layers:
                module = layer.block_sparse_moe.experts[expert_id]
                w1 = _dequantize_awq_linear_to_fp32(module.w1)
                w3 = _dequantize_awq_linear_to_fp32(module.w3)
                w2 = _dequantize_awq_linear_to_fp32(module.w2)
                w13 = torch.cat([w1, w3], dim=0).contiguous()
                per_layer_w13.append((w13, 1.0))
                per_layer_w2.append((w2, 1.0))
            cpu_pool[expert_id] = _ShadowExpertINT4(per_layer_w13, per_layer_w2)
        return cpu_pool

    def _refresh_hot_cold_classification(self) -> None:
        """Issue #33 Fase 3 — ricalcola self._hot_expert_ids da hotness EAT
        reale, via SEEPolicy.classify_hot_cold() (Fase 0).

        Chiamata solo qui e in refresh_shadow_pool_selection() — NON ad
        ogni route_forward() (potenzialmente una volta per token):
        tier_manager.eat.get_tier(Tier.DDR4) scansiona l'intera tabella,
        stesso motivo per cui _select_shadow_expert_ids()/_shadow_pool
        stesso sono ricalcolati solo al load/refresh esplicito, non ad
        ogni forward — vedi il commento gemello lì.

        Fallback onesto a cold start (nessuna entry EAT ancora, o
        self._tier_manager assente): TUTTI gli expert_id del pool restano
        "caldi" — comportamento pre-Fase-3 invariato (GPU-only) invece di
        instradare tutto a freddo su un segnale che non esiste ancora.
        Stesso principio già usato da _select_shadow_expert_ids() per il
        suo fallback round-robin.
        """
        tier_manager = getattr(self, "_tier_manager", None)
        entries = tier_manager.eat.get_tier(Tier.DDR4) if tier_manager is not None else []
        if not entries:
            self._hot_expert_ids = set(self._shadow_pool) | set(self._cpu_shadow_pool)
            return
        # getattr difensivo — stesso motivo di _tier_manager sopra: un
        # worker di test costruito via __new__() prima di questa
        # integrazione (issue #33 Fase 3) non ha _hot_cold_policy assegnato.
        policy = getattr(self, "_hot_cold_policy", None) or SEEPolicy()
        hot_ids, _cold_ids = policy.classify_hot_cold(entries)
        self._hot_expert_ids = set(hot_ids)

    def route_forward(self, expert_id: int, layer_id: int, hidden_states: Any) -> Any:
        """Issue #33 Fase 3 — dispatcha la chiamata shadow per expert_id al
        pool GPU (self._shadow_pool, VRAM) o al pool CPU/DDR4-resident
        (self._cpu_shadow_pool, Fase 2), secondo la classificazione hot/cold
        più recente (self._hot_expert_ids, Fase 0 via
        _refresh_hot_cold_classification()).

        Il Tier enum NON viene toccato qui — un expert instradato a freddo
        resta logicamente Tier.DDR4 in EAT, come richiesto dal piano
        originale: questo metodo sceglie solo quale callable eseguire, non
        modifica alcuno stato di tiering. Un expert_id "freddo" non
        attraversa mai GPUTransfer.to_vram()/TierManager.promote_live_tensor()
        per questa chiamata — self._cpu_shadow_pool è costruito da
        _build_cpu_shadow_pool() (Fase 2), che non chiama mai .to() in
        nessuna direzione (vedi il suo docstring).

        Fallback quando un expert_id non è in entrambi i pool (es. path
        2/3 — Marlin/AWQ — dove Fase 2 non costruisce un pool CPU, o un
        expert presente solo in uno dei due per qualunque altro motivo):
        usa qualunque pool lo contenga, ignorando la classificazione —
        meglio un forward funzionante nell'unica residenza disponibile che
        nessun forward.

        Args:
            expert_id:     Expert da eseguire (deve essere presente in
                           almeno uno dei due pool — non verificato qui,
                           è responsabilità del chiamante, stesso
                           contratto di un dict semplice indicizzato con
                           una chiave assente).
            layer_id:      Layer corrente (un "expert i" ha pesi diversi
                           per layer).
            hidden_states: Input del forward per il layer corrente.

        Returns:
            Output del forward, dalla residenza scelta.

        BUG REALE (2026-08-17, trovato scrivendo benchmarks/bench_hybrid.py
        — Fase 5, non un unit test isolato): hidden_states nel path reale
        arriva dalla forward pass del modello, quindi è CUDA-resident (il
        forward hook `.gate` gira dentro un modello che vive su GPU) — un
        expert instradato a freddo lo passava intatto a pesi CPU-resident,
        stesso "Expected all tensors to be on the same device" del bug
        2026-08-12 sui PESI (vedi commento nel loop path 1 di
        _load_shadow_pool()), stavolta sull'INPUT. Nessun test di Fase 2/3
        l'aveva mai esercitato: usavano sempre hidden_states CPU-resident
        per costruzione (stesso device dei pesi CPU del test). Fix: sposta
        hidden_states su CPU immediatamente prima della sola chiamata al
        pool CPU — un D2H di un batch a una riga, non dell'intero modello,
        stesso costo che qualunque altro compute-offload CPU pagherebbe.
        """
        # getattr difensivo — stesso motivo di _RoutedShadowPool.__contains__:
        # un worker di test pre-Fase-2/3 non ha _cpu_shadow_pool/
        # _hot_expert_ids assegnati.
        cpu_pool = getattr(self, "_cpu_shadow_pool", {})
        hot_expert_ids = getattr(self, "_hot_expert_ids", set())
        in_gpu_pool = expert_id in self._shadow_pool
        in_cpu_pool = expert_id in cpu_pool

        route_to_cpu = (
            (in_gpu_pool and in_cpu_pool and expert_id not in hot_expert_ids)
            or (in_cpu_pool and not in_gpu_pool)
        )
        if route_to_cpu:
            if hidden_states.device.type != "cpu":
                hidden_states = hidden_states.cpu()
            return cpu_pool[expert_id](hidden_states, layer_id)
        return self._shadow_pool[expert_id](hidden_states, layer_id)

    # ── M1/M2 wiring (2026-08-12, issue #17) ────────────────────────────────────

    def _seed_eat_entries(self) -> None:
        """Inserisce in EAT una entry (expert_id, layer_id) per OGNI
        combinazione reale del modello caricato, a Tier.DDR4.

        DDR4, non NVME: questi pesi vivono già sull'host per costruzione
        (residenti in GPU o offloaded su CPU da vLLM stesso), mai su un
        file NVMe separato — DDR4 è la tier di partenza onesta, non un
        placeholder scelto per comodità.

        TUTTI gli expert, non solo i shadow_pool_size attualmente in pool:
        senza, EAT.hottest_candidates()/get_tier() non potrebbe mai
        scoprire un expert diverso da quello già in pool, e la selezione
        sarebbe round-robin travestita da hotness-driven — vedi
        _select_shadow_expert_ids().

        Idempotente (eat.lookup() prima di ogni insert()): sicuro da
        richiamare più volte nello stesso processo.
        """
        model = self._base.model_runner.model
        layers = model.model.layers
        first_experts = layers[0].block_sparse_moe.experts
        is_fused = hasattr(first_experts, "num_experts")
        n_experts = first_experts.num_experts if is_fused else len(first_experts)
        n_layers = len(layers)

        eat = self._tier_manager.eat
        seeded = 0
        for expert_id in range(n_experts):
            for layer_id in range(n_layers):
                if eat.lookup(expert_id, layer_id) is None:
                    eat.insert(expert_id, layer_id, tier=Tier.DDR4, size_bytes=0)
                    seeded += 1
        log.info(
            "GCSG: EAT seeded — %d entry (%d expert x %d layer) a Tier.DDR4.",
            seeded, n_experts, n_layers,
        )

    def _select_shadow_expert_ids(self, n_experts: int) -> list[int]:
        """Seleziona quali expert_id popolano lo shadow pool.

        Senza TierManager (default): round-robin placeholder invariato
        (range(...)) — comportamento identico a prima di questa
        integrazione.

        Con TierManager: aggrega EAT.get_tier(Tier.DDR4) per expert_id
        (somma access_count su tutti i layer di quell'expert) e prende i
        primi shadow_pool_size per punteggio decrescente. Tie-break
        DELIBERATAMENTE sul solo ordine di iterazione (range(n_experts),
        ascendente), non su last_access_ts: quest'ultimo sembrava
        un'opzione più "intelligente" ma è un falso segnale a freddo — a
        entry appena seeded da _seed_eat_entries() (stesso access_count
        globale, zero traffico reale), last_access_ts riflette solo
        l'ordine di inserimento nel loop di seeding, non hotness reale, e
        avrebbe silenziosamente favorito l'ultimo expert_id seeded invece
        di comportarsi da round-robin (bug trovato scrivendo i test di
        questa stessa funzione, prima che arrivasse su hardware reale).
        sorted(..., reverse=True) è stabile in Python (garantito dalla
        documentazione — reverse non rompe la stabilità), quindi a parità
        di punteggio (incluso il caso AL PRIMO LOAD, tutte le entry a
        access_count=0) l'ordine di input range(n_experts) sopravvive
        intatto: equivalente esatto al round-robin — proprietà onesta di
        un cold start, non un difetto nascosto: non esiste segnale di
        hotness prima che un solo token sia stato instradato. Il valore
        reale di questo path emerge chiamando refresh_shadow_pool_selection()
        dopo che EAT ha accumulato traffico reale da
        _evaluate_gcsg_for_rows() — vedi la sua docstring per perché non è
        (ancora) agganciata a un trigger automatico.
        """
        pool_size = min(self.guard.shadow_pool_size, n_experts)
        # getattr difensivo, non self._tier_manager diretto: un GCSGWorker
        # costruito via __new__() nei test (bypassando __init__(), stesso
        # pattern usato altrove in questo file/nei test) non ha _base né
        # _tier_manager finché non assegnati a mano — un accesso diretto
        # cadrebbe in __getattr__, che delega a self._base, anch'esso
        # assente in quel caso -> RecursionError, non un AttributeError
        # pulito. Stesso motivo per cui _current_row_request_ids sotto usa
        # già getattr(..., None).
        tier_manager = getattr(self, "_tier_manager", None)
        if tier_manager is None:
            return list(range(pool_size))

        entries = tier_manager.eat.get_tier(Tier.DDR4)
        if not entries:
            # EAT non seeded (es. _seed_eat_entries fallita in load_model,
            # già loggato lì) — stesso fallback del caso senza TierManager.
            return list(range(pool_size))

        scores: dict[int, int] = {}
        for entry in entries:
            scores[entry.expert_id] = scores.get(entry.expert_id, 0) + entry.access_count
        ranked = sorted(range(n_experts), key=lambda e: scores.get(e, 0), reverse=True)
        return ranked[:pool_size]

    def refresh_shadow_pool_selection(self) -> None:
        """Ricalcola la selezione dello shadow pool da EAT hotness reale e
        ricarica il pool se cambia.

        Pensato per essere chiamato periodicamente (es. ogni N richieste)
        una volta che il costo reale di promote()/evict() è stato
        misurato su hardware vero — sotto-obiettivo 4 dello Sprint 4
        ("shard promotion latency"). Deliberatamente NON agganciato a
        nessun trigger automatico in questa integrazione: farlo prima di
        avere quel dato rischierebbe di introdurre uno storm di
        promote/evict a ogni chiamata, esattamente il costo che quel
        sotto-obiettivo deve prima quantificare, non assumere trascurabile.
        No-op silenzioso se non wired a un TierManager, o se la selezione
        non cambia.
        """
        if getattr(self, "_tier_manager", None) is None:
            log.warning(
                "GCSG: refresh_shadow_pool_selection() no-op — nessun "
                "TierManager wired (shadow pool round-robin, nulla da "
                "aggiornare)."
            )
            return
        if getattr(self, "_n_experts_cached", None) is None:
            log.warning(
                "GCSG: refresh_shadow_pool_selection() chiamato prima di "
                "_load_shadow_pool() — no-op."
            )
            return

        new_ids = self._select_shadow_expert_ids(self._n_experts_cached)
        if sorted(new_ids) == sorted(self._shadow_pool.keys()):
            return
        log.info(
            "GCSG: refresh_shadow_pool_selection — nuova selezione %s (era "
            "%s), ricarico lo shadow pool.",
            new_ids, sorted(self._shadow_pool.keys()),
        )
        self._shadow_pool.clear()
        # getattr difensivo — stesso motivo di _tier_manager/_n_experts_cached
        # sopra: un GCSGWorker di test costruito via __new__() prima di questa
        # integrazione (issue #33 Fase 2) non ha _cpu_shadow_pool assegnato.
        cpu_pool = getattr(self, "_cpu_shadow_pool", None)
        if cpu_pool is not None:
            cpu_pool.clear()
        self._load_shadow_pool()

    def _should_pin_transfers(self) -> bool:
        """True su Linux reale (dove il soak test 2026-08-12 ha verificato
        pinning sicuro sotto carico sostenuto — vedi LOGBOOK.md e GCSG
        report §9), False sotto WSL2 (mai validato sotto carico sostenuto
        lì). Stessa funzione in_wsl() già usata altrove nel progetto per
        questa identica decisione. Conservativo (False) se non
        determinabile — es. vllm non importabile in questo processo, non
        dovrebbe succedere qui ma non è un'assunzione su cui vale la pena
        fallire rumorosamente.
        """
        try:
            from vllm.platforms.interface import in_wsl
            return not in_wsl()
        except Exception:
            return False

    def _promote_module_via_tier_manager(self, module: Any, expert_id: int, layer_id: int) -> None:
        """Promuove i Parameter CPU-resident di `module` in VRAM via
        TierManager, sostituendoli in-place (stesso effetto finale di
        expert.to('cuda'), che questo sostituisce nel path AWQ
        ModuleList quando self._tier_manager è wired).

        Granularità: UN solo Parameter per (expert_id, layer_id) viene
        tracciato in EAT/TierManager — quello con più elementi (dominante
        per peso in un modulo AWQ-packed, tipicamente qweight) — non ogni
        singolo tensore del modulo. Gli altri Parameter (es.
        qzeros/scales, ordini di grandezza più piccoli) vengono comunque
        spostati realmente su GPU con la stessa decisione di pinning, ma
        via una copia diretta non tracciata singolarmente in EAT:
        SHARD_SIZE_MB=256 e tutta la contabilità EAT sono pensati per
        asset a grana di peso principale (vedi memory math nel docstring
        di modulo), non per array di scale/zero-point — tracciarli uno
        per uno moltiplicherebbe le entry EAT senza segnale utile in più.

        NON verificato su hardware reale (nessuna GPU in questo ambiente):
        il transfer stesso è verificato a livello di GPUTransfer.to_vram()
        (TestGPUTransfer, self-hosted runner) e TierManager.promote_live_tensor()
        è pura logica Python + quella stessa chiamata — ma il bridging
        asyncio.run() dentro load_model() (chiamato sync dal worker vLLM,
        un processo dedicato senza event loop già attivo per quanto
        verificato nella sequenza di avvio in
        vllm.executor.gpu_executor.GPUExecutor — stessa fonte già usata
        per l'ordine init_device/load_model, non però per QUESTO specifico
        bridging) va confermato sul pod. Primo item da controllare.
        """
        import asyncio

        pin = self._should_pin_transfers()
        cpu_named = [
            (name, p) for name, p in module.named_parameters()
            if p.device.type != "cuda"
        ]
        if not cpu_named:
            return

        dominant_name, dominant_param = max(cpu_named, key=lambda np_: np_[1].numel())
        vram_tensor = asyncio.run(
            self._tier_manager.promote_live_tensor(
                expert_id, layer_id, dominant_param.data, pin=pin,
            )
        )
        dominant_param.data = vram_tensor

        for name, param in cpu_named:
            if name == dominant_name:
                continue
            cpu_tensor = param.data.pin_memory() if pin else param.data
            param.data = cpu_tensor.to("cuda", non_blocking=pin)

    def _pin_awq_expert_to_gpu(
        self, layers: list[Any], expert_id: int,
    ) -> list[Any] | None:
        """Assicura che expert_id sia residente in GPU su TUTTE le layer,
        spostandolo esplicitamente se offloaded — copia sincrona reale
        (.to('cuda'), NON il .to(..., non_blocking=True) di
        maybe_offload_to_cpu, che comunque non tocca mai i singoli moduli
        MixtralMLP, solo l'intera decoder layer — vedi issue #16).
        expert.forward non è mai wrappato da vLLM (verificato,
        scripts/map_offload_state.py): nessun forward da "ripristinare",
        basta che i suoi Parameter siano su GPU prima della prima chiamata.

        Con self._tier_manager wired (2026-08-12, issue #17): il transfer
        passa da _promote_module_via_tier_manager() invece di un
        .to('cuda') diretto — stessa destinazione fisica finale, ma ora
        tracciato in EAT (tier VRAM) attraverso il punto di controllo di
        M2 invece di un side-channel separato. self._tier_manager is None
        (default): comportamento invariato, .to('cuda') diretto come
        prima di questa integrazione.

        Ritorna la lista dei moduli (uno per layer) se il pinning riesce su
        TUTTE le layer, None se anche una sola fallisce — l'expert_id intero
        resta fuori dal pool, non un mix di layer pinnate/non pinnate che
        degraderebbe in modo silenzioso e imprevedibile a metà forward.
        """
        modules = []
        for layer_id, layer in enumerate(layers):
            expert = layer.block_sparse_moe.experts[expert_id]
            try:
                if next(expert.parameters()).device.type != "cuda":
                    if getattr(self, "_tier_manager", None) is not None:
                        self._promote_module_via_tier_manager(expert, expert_id, layer_id)
                    else:
                        expert.to("cuda")
            except Exception as e:
                log.warning(
                    "GCSG: impossibile pinnare l'expert AWQ %d (layer %d) in "
                    "GPU (%s) — escluso dallo shadow pool.",
                    expert_id, layer_id, e,
                )
                return None
            modules.append(expert)
        return modules

    def _build_marlin_shadow_pool(
        self, layers: list[Any], expert_ids: list[int],
    ) -> dict[int, _MarlinFusedShadowExpert]:
        """Costruisce le entry per-layer (fused, expert_id_locale_o_globale,
        num_experts) che _MarlinFusedShadowExpert consuma, condivise da
        TUTTI gli expert_ids insieme (un proxy per layer OFFLOADED, non uno
        per expert_id — raddoppierebbe il costo VRAM per nulla dato che
        _MarlinFusedShadowExpert isola comunque un solo target per chiamata
        via router_logits).

        IMPORTANTE (2026-08-10, corretto dopo un hang — vedi ATTENZIONE su
        _PinnedMarlinExperts): _PinnedMarlinExperts viene costruito SOLO per
        le layer effettivamente offloaded (w13_qweight.device.type=="cpu").
        Per le altre — la maggioranza, già GPU-resident — si riusa il modulo
        FusedMoE originale direttamente, num_experts=8 reale, expert_id
        GLOBALE: zero copie, zero allocazioni extra, nessun rischio di
        frammentazione. Il primo tentativo proxava indiscriminatamente
        tutte e 32 le layer e ha bloccato il profiling VRAM di vLLM.

        Tutto o niente: se il pinning fallisce su anche una sola layer
        offloaded, NESSUN expert Marlin entra nel pool (i proxy sono
        condivisi tra tutti gli expert_ids del pool — un fallimento
        parziale lascerebbe alcuni _MarlinFusedShadowExpert con layer
        mancanti in modo silenzioso).

        Wiring TierManager (2026-08-12, issue #17 — deciso con l'utente
        di procedere una volta che il path AWQ fosse stato verificato due
        volte su hardware reale con determinismo perfetto): quando
        self._tier_manager è wired, ogni _PinnedMarlinExperts costruito
        qui riceve un tensor_promoter (vedi _build_marlin_tensor_promoter())
        che instrada UN tensore dominante per layer (w13_qweight)
        attraverso TierManager.promote_live_tensor() invece di un
        .to(device) diretto — gli altri cinque tensori per layer si
        muovono con la stessa decisione di pinning ma non tracciati
        singolarmente in EAT, stesso principio già usato per il path AWQ
        (_promote_module_via_tier_manager). self._tier_manager is None
        (default): comportamento invariato, .to(device) diretto come
        prima di questa integrazione — zero rischio per il path già
        validato.
        """
        # entries_per_expert[expert_id][layer_id] = (fused, expert_id_in_fused, num_experts_in_fused)
        entries_per_expert: dict[int, list[tuple[Any, int, int] | None]] = {
            expert_id: [None] * len(layers) for expert_id in expert_ids
        }

        for layer_id, layer in enumerate(layers):
            experts = layer.block_sparse_moe.experts
            is_offloaded = experts.w13_qweight.data.device.type == "cpu"

            if not is_offloaded:
                for expert_id in expert_ids:
                    entries_per_expert[expert_id][layer_id] = (
                        experts, expert_id, experts.num_experts,
                    )
                continue

            try:
                promoter = (
                    self._build_marlin_tensor_promoter(layer_id, expert_ids)
                    if getattr(self, "_tier_manager", None) is not None else None
                )
                proxy = _PinnedMarlinExperts(experts, expert_ids, tensor_promoter=promoter)
            except Exception as e:
                log.warning(
                    "GCSG: impossibile pinnare gli expert Marlin %s (layer %d, "
                    "offloaded) in GPU (%s) — path Marlin escluso dallo shadow "
                    "pool.", expert_ids, layer_id, e,
                )
                return {}
            for local_index, expert_id in enumerate(expert_ids):
                entries_per_expert[expert_id][layer_id] = (proxy, local_index, len(expert_ids))

        return {
            expert_id: _MarlinFusedShadowExpert(entries)
            for expert_id, entries in entries_per_expert.items()
        }

    def _build_marlin_tensor_promoter(
        self, layer_id: int, expert_ids: list[int],
    ) -> Callable[[str, Any], Any]:
        """Costruisce la funzione di transfer che _PinnedMarlinExperts usa
        al posto di `.to(device)` diretto quando self._tier_manager è
        wired (2026-08-12, issue #17).

        UN tensore dominante per layer (w13_qweight — il più grande, per
        analogia con la scelta già fatta per il path AWQ) passa da
        TierManager.promote_live_tensor(); gli altri cinque
        (w2_qweight/w13_scales/w2_scales/w13_qzeros/w2_qzeros) si
        muovono con la stessa decisione di pinning ma senza tracciamento
        EAT individuale — stesso principio di
        _promote_module_via_tier_manager() per il path AWQ, vedi la sua
        docstring per il perché (SHARD_SIZE_MB e tutta la contabilità EAT
        sono pensati per asset a grana di peso principale, non per array
        di scale/zero-point).

        Chiave EAT: expert_id=-1 (sentinella — mai un ID expert reale),
        shard_idx = _marlin_pool_shard_key(layer_id, expert_ids), NON
        solo layer_id — vedi la sua docstring per il perché quella
        distinzione è necessaria qui e non lo è per il path AWQ (dove la
        chiave è già per-singolo-expert, non condivisa da un pool).

        BUG REALE trovato sul pod (2026-08-12, prima verifica hardware di
        questo path): `_build_marlin_shadow_pool()` decide "offloaded"
        controllando SOLO `w13_qweight.data.device.type` — assunzione
        implicita (mai vera per costruzione) che le altre cinque tensori
        Marlin-packed dello stesso layer condividano lo stesso device.
        Non è così: su hardware reale, `w13_scales`/`w2_scales`/altre
        possono restare GPU-resident anche quando `w13_qweight` è
        offloaded su CPU — cpu_offload_gb di vLLM evidentemente non
        offload'a l'intero modulo come unità indivisibile. Il ramo non-
        dominante sotto chiamava `.pin_memory()` incondizionatamente,
        che solleva `RuntimeError: cannot pin ... only dense CPU tensors
        can be pinned` su un tensore già CUDA — esattamente l'errore
        osservato. Fix: controllo esplicito del device PRIMA di
        pinnare/spostare, stesso principio difensivo già usato per il
        path AWQ (_promote_module_via_tier_manager() filtra
        `p.device.type != "cuda"` prima di processare ogni parametro —
        qui manca lo stesso controllo, aggiunto ora). Il ramo dominante
        (via TierManager.promote_live_tensor() -> GPUTransfer.to_vram())
        era già sicuro per costruzione: to_vram() riporta esplicitamente
        su CPU un input già CUDA prima di un eventuale pin_memory() —
        vedi tier/gpu.py.
        """
        import asyncio

        pin = self._should_pin_transfers()
        shard_idx = self._marlin_pool_shard_key(layer_id, expert_ids)
        dominant_name = "w13_qweight"

        def _promoter(name: str, cpu_slice: Any) -> Any:
            if cpu_slice.device.type == "cuda":
                # Già GPU-resident — nulla da promuovere/pinnare, stesso
                # comportamento del vecchio .to(device) diretto (no-op su
                # un tensore già sul device target). Vedi BUG REALE sopra.
                return cpu_slice
            if name == dominant_name:
                return asyncio.run(
                    self._tier_manager.promote_live_tensor(
                        expert_id=-1, shard_idx=shard_idx, cpu_data=cpu_slice, pin=pin,
                    )
                )
            return cpu_slice.pin_memory().to("cuda", non_blocking=True) if pin else cpu_slice.to("cuda")

        return _promoter

    @staticmethod
    def _marlin_pool_shard_key(layer_id: int, expert_ids: list[int]) -> int:
        """Chiave shard_idx sentinella per la entry EAT del proxy Marlin
        condiviso di un layer (2026-08-12, issue #17).

        Dipende SIA da layer_id SIA dalla composizione del pool
        (`hash(tuple(sorted(expert_ids)))`), non solo da layer_id: questo
        proxy è condiviso da TUTTI gli expert_ids del pool insieme (mai
        un proxy per singolo expert_id — raddoppierebbe il costo VRAM per
        nulla, vedi _PinnedMarlinExperts). Se la chiave dipendesse solo
        da layer_id, un refresh_shadow_pool_selection() che cambiasse la
        composizione del pool per lo stesso layer riutilizzerebbe per
        errore il tensore VRAM della composizione precedente —
        promote_live_tensor() è idempotente per chiave by design (corretto
        per il caso normale per-expert del path AWQ, dove ogni expert_id
        ha sempre e solo i propri dati), pericoloso qui senza questo
        accorgimento perché la STESSA chiave altrimenti rappresenterebbe
        dati fisicamente diversi a seconda di quali expert_ids sono nel
        pool in quel momento.

        hash() su una tupla di int è deterministico entro lo stesso
        processo — non soggetto a PYTHONHASHSEED, che randomizza solo
        l'hashing di str/bytes, non di tuple di interi piccoli. Non serve
        stabilità cross-processo: ogni processo GCSGWorker ha il proprio
        TierManager/EAT, mai condivisi tra processi.
        """
        return layer_id * 1_000_000_007 + (hash(tuple(sorted(expert_ids))) % 1_000_000_007)

    def _register_gate_hooks(self) -> None:
        """Forward hook su ogni layer_i.block_sparse_moe.gate — cattura
        router_logits E hidden_states reali, poi valuta GCSG per-riga.

        Verificato: MixtralMoE.forward() calcola router_logits internamente
        ma non lo restituisce; self.gate (ReplicatedLinear) sì, nel suo
        stesso output. Un hook sul blocco MoE intero vedrebbe solo l'hidden
        state finale già ricombinato dagli expert, non i gating score.
        inputs[0] dell'hook È l'hidden_states passato a self.gate(x) — stesso
        tensore usato dagli expert reali, qui riusato per lo shadow forward.
        """
        model = self._base.model_runner.model
        for layer_id, layer in enumerate(model.model.layers):
            gate = layer.block_sparse_moe.gate

            def _capture_and_evaluate(module, inputs, output, _worker=self, _layer_id=layer_id):
                router_logits, _bias = output
                hidden_states = inputs[0]
                # Tetto a _MAX_CAPTURED_ROUTER_LOGITS (2026-08-10, issue #10/#16):
                # senza, questa lista cresce senza limite, un tensore GPU per hit,
                # per l'intera vita del processo — vedi il commento su
                # self.captured_router_logits in __init__ per la cronologia del
                # bug (crescita illimitata -> pressione/frammentazione memoria GPU
                # sui carichi reali, mai vista sugli smoke test che l'hanno scritta).
                if len(_worker.captured_router_logits) < _worker._MAX_CAPTURED_ROUTER_LOGITS:
                    _worker.captured_router_logits.append(router_logits.detach())
                _worker._evaluate_gcsg_for_rows(router_logits, hidden_states, _layer_id)
                return output

            handle = gate.register_forward_hook(_capture_and_evaluate)
            self._gate_hook_handles.append(handle)

    def _evaluate_gcsg_for_rows(self, router_logits: Any, hidden_states: Any, layer_id: int) -> None:
        """Da router_logits/hidden_states reali (per riga = per token) a
        GatingContext, poi should_activate_shadow()/run_shadow() — con
        request_id reali (da execute_model(), via
        _current_row_request_ids) e hidden_states reali (da questo stesso
        hook — vedi _register_gate_hooks).

        Skip silenzioso se il numero di righe non combacia con
        _current_row_request_ids: succede durante il profile_run() di vLLM
        (chiamato da determine_num_available_blocks(), non da execute_model())
        con dati sintetici e nessun request_id reale da associare — non un
        errore, solo un forward pass che GCSG non deve valutare.
        """
        import math

        import torch

        row_request_ids = getattr(self, "_current_row_request_ids", None)
        if not row_request_ids or len(row_request_ids) != router_logits.shape[0]:
            return

        probs = torch.softmax(router_logits.float(), dim=-1)
        n_experts = probs.shape[-1]
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1) / math.log(n_experts)

        # Batched device->host sync (2026-08-10, candidate mechanism for a
        # reproducible stall under certain batch compositions — see LOGBOOK):
        # previously .tolist()/float() ran INSIDE the loop below, one CUDA sync
        # pair per row. Every .gate hook call (every layer, every forward pass)
        # paid that N times — 32 layers x batch_size blocking syncs per token,
        # scaling with batch size. Hoisted out of the loop: same data, same
        # GatingContext fields, same should_activate_shadow()/run_shadow()
        # logic below, just two syncs total per hook call instead of 2xN.
        gating_scores_batch = probs.tolist()
        entropy_batch = entropy.tolist()
        # getattr difensivo (non self._tier_manager diretto): vedi il
        # commento gemello in _select_shadow_expert_ids sul perché — un
        # GCSGWorker costruito via __new__() senza _base impostato
        # ricadrebbe in RecursionError altrimenti. Hoisted fuori dal loop
        # per riga: un solo lookup per hook call, non uno per token.
        tier_manager = getattr(self, "_tier_manager", None)

        for row_idx, request_id in enumerate(row_request_ids):
            row_scores = gating_scores_batch[row_idx]
            ctx = GatingContext(
                token_id=row_idx,
                request_id=request_id,
                gating_scores=row_scores,
                token_entropy=entropy_batch[row_idx],
            )
            if tier_manager is not None:
                # Traffico EAT reale (2026-08-12, issue #17): il top-1 qui
                # è la VERA decisione di routing MoE per questo token/layer
                # — indipendente da should_activate_shadow()/run_shadow()
                # sotto, che decidono solo se GCSG interviene. Senza questo,
                # EAT non vedrebbe mai traffico reale (solo il conteggio di
                # attivazioni shadow, un sottoinsieme molto più piccolo e
                # distorto verso i token ad alta confidenza — vedi
                # GCSGGuard.should_activate_shadow) — è esattamente la
                # "traffico concorrente reale" che issue #1/#2/#4 (M1,
                # sotto-obiettivo 5) hanno bisogno di misurare.
                top1_expert_id = max(range(len(row_scores)), key=row_scores.__getitem__)
                tier_manager.eat.access(top1_expert_id, layer_id)
            should, _ = self.guard.should_activate_shadow(ctx)
            if should:
                # hidden_states[row_idx:row_idx+1], NON hidden_states[row_idx]
                # (2026-08-10, bug reale trovato dal primo run MMLU con shadow
                # execution davvero attiva — mai esercitato prima, la shadow
                # execution non aveva mai raggiunto questo punto): indicizzare
                # con uno scalare collassa la dimensione riga, producendo un
                # tensore 1D (hidden_dim,) invece di un batch a una riga
                # (1, hidden_dim). _MarlinFusedShadowExpert costruisce
                # router_logits da hidden_states.shape[0] assumendo 2D — con
                # input 1D usa hidden_dim al posto del numero di righe,
                # crashando più a valle in FusedMoE.select_experts()
                # ("not enough values to unpack (expected 2, got 1)"). Lo
                # slicing con range preserva la forma 2D — stesso principio
                # per cui _AWQShadowExpert/_ShadowExpertINT4 non erano mai
                # andati in crash: i loro matmul tollerano un input 1D via
                # broadcasting, producendo silenziosamente un output 1D
                # anziché sollevare un errore — sbagliato allo stesso modo,
                # solo senza un crash che lo segnalasse.
                # _RoutedShadowPool(self) (issue #33 Fase 3), non
                # self._shadow_pool diretto: decide per-expert se
                # instradare a GPU o al pool CPU/DDR4-resident (Fase 2)
                # senza che run_shadow() debba saperne nulla — vedi
                # _RoutedShadowPool/route_forward(). Costruito qui invece
                # che cacheato in __init__: oggetto stateless (una sola
                # referenza a self), zero costo reale, e non richiede che
                # ogni worker di test costruito via __new__() lo assegni
                # a mano.
                self.guard.run_shadow(
                    ctx, _RoutedShadowPool(self),
                    hidden_states=hidden_states[row_idx : row_idx + 1], layer_id=layer_id,
                )

    def execute_model(self, *args, **kwargs):
        """Delega a Worker.execute_model, dopo aver preparato la riga -> request_id
        mapping che gli hook .gate consumano durante la chiamata nested sotto.

        CORREZIONE (smoke test 2026-08-09): questo override eredita da
        WorkerBase.execute_model, che prende un ExecuteModelRequest — NON lo
        stesso execute_model() di ModelRunner (model_input/kv_caches/...),
        nome uguale ma parametro diverso, confuso inizialmente scrivendo
        questa classe. request_ids_to_seq_ids NON esiste su
        ExecuteModelRequest (verificato: dir() non lo elenca) — i request_id
        reali si estraggono da execute_model_req.seq_group_metadata_list
        (List[SequenceGroupMetadata], ognuno con .request_id), confermato
        via un run reale con logging di debug.

        La decisione GCSG vera e propria (should_activate_shadow/run_shadow
        con gating_scores/hidden_states reali) avviene dentro
        _evaluate_gcsg_for_rows(), chiamata dagli hook .gate — qui si prepara
        solo _current_row_request_ids: ogni SequenceGroupMetadata contribuisce
        token_chunk_size righe, nello stesso ordine in cui ModelRunner
        concatena le sequenze in un'unica hidden_states batched (verificato
        empiricamente: la lunghezza combacia sempre con router_logits.shape[0]
        quando la richiesta viene da execute_model(), mai durante il
        profile_run() sintetico di vLLM — vedi _evaluate_gcsg_for_rows).
        """
        execute_model_req = args[0] if args else kwargs.get("execute_model_req")
        seq_group_metadata_list = getattr(execute_model_req, "seq_group_metadata_list", None) or []

        self.seen_request_ids.update(smd.request_id for smd in seq_group_metadata_list)
        self._current_row_request_ids = [
            smd.request_id
            for smd in seq_group_metadata_list
            for _ in range(smd.token_chunk_size)
        ]

        return self._base.execute_model(*args, **kwargs)
