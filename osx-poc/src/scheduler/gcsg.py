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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
import time

import pynvml   # provided by the `nvidia-ml-py` package (requirements.txt) —
                # the standalone `pynvml` PyPI package is deprecated, this
                # is the same import name from NVIDIA's maintained bindings

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
    gating_scores:  List[float]     # score per ogni expert (len = n_experts)
    token_entropy:  float
    bf16_available: bool = False    # deciso dal chiamante (Tier Manager/EAT);
                                     # GCSGGuard non lo deduce da solo
    timestamp:      float = field(default_factory=time.monotonic)


@dataclass
class ShadowExecutionResult:
    """Risultato di una shadow execution."""
    activated:          bool           # shadow execution eseguita?
    shadow_expert_id:   Optional[int]  # expert INT4 usato
    contamination_flag: bool           # questo token è "shadow-contaminated"?
    latency_ms:         float          # overhead shadow execution
    reason_skip:        Optional[str]  # perché shadow non attivata (debug)


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
        self._request_tokens:        Dict[str, int] = {}   # request_id -> token valutati
        self._request_contamination: Dict[str, int] = {}   # request_id -> token shadow-contaminati

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

    def should_activate_shadow(self, ctx: GatingContext) -> Tuple[bool, str]:
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
        shadow_pool: Dict[int, object],
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

    def contamination_rate(self, request_id: Optional[str] = None) -> float:
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

    def reset_contamination_counter(self, request_id: Optional[str] = None) -> None:
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

    def update_thresholds(self, theta_gate: Optional[float] = None,
                          theta_entropy: Optional[float] = None,
                          theta_contamination: Optional[float] = None) -> None:
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


def _quantize_int4(weight: Any) -> Tuple[Any, float]:
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
    """

    def __init__(
        self,
        per_layer_w13: List[Tuple[Any, float]],
        per_layer_w2: List[Tuple[Any, float]],
    ) -> None:
        self._per_layer_w13 = per_layer_w13
        self._per_layer_w2 = per_layer_w2

    def __call__(self, hidden_states: Any, layer_id: int) -> Any:
        import torch.nn.functional as F

        w13_q, w13_scale = self._per_layer_w13[layer_id]
        w2_q, w2_scale = self._per_layer_w2[layer_id]
        w13 = w13_q.to(hidden_states.dtype) * w13_scale
        w2 = w2_q.to(hidden_states.dtype) * w2_scale

        intermediate_size = w2.shape[-1]
        gate_up = hidden_states @ w13.T
        gate, up = gate_up.split(intermediate_size, dim=-1)
        activated = F.silu(gate) * up
        return activated @ w2.T


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
    placeholder sintetici in execute_model(). NON ancora riverificato con
    un secondo smoke test end-to-end dopo queste modifiche — farlo è il
    prossimo passo prima di fidarsene in produzione.

    L'import di vllm è locale ai metodi (non al modulo) apposta: gcsg.py deve
    restare importabile — e GCSGGuard testabile — anche in ambienti senza
    vLLM installato (es. CI cpu-tests, che non installa requirements-vllm.txt).
    """

    def __init__(self, *args, guard: Optional[GCSGGuard] = None, **kwargs) -> None:
        from vllm.worker.worker import Worker   # import locale, vedi docstring classe
        self._base = Worker(*args, **kwargs)
        self.guard = guard or GCSGGuard()
        self._shadow_pool: Dict[int, object] = {}
        self._gate_hook_handles: List[object] = []

        # Osservabilità smoke-test (2026-08-09) — non usata dal path di
        # produzione, permette di verificare dall'esterno che gli hook
        # sparino davvero e che i request_id reali arrivino a execute_model().
        self.captured_router_logits: List[object] = []   # torch.Tensor per hit, non tipizzato qui per non importare torch al modulo
        self.seen_request_ids: set = set()

        # riga -> request_id per il batch corrente, popolato da execute_model()
        # e consumato dagli hook .gate (_evaluate_gcsg_for_rows) durante la
        # chiamata nested — vedi execute_model().
        self._current_row_request_ids: List[str] = []

    def __getattr__(self, name):
        # Delega tutto ciò che non sovrascriviamo esplicitamente al Worker reale
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
        try:
            self._load_shadow_pool()
        except Exception as e:
            log.warning(
                "GCSG: shadow pool non caricato (%s) — GCSGWorker gira in "
                "modalità hook-only: hook/request_id/contamination bookkeeping "
                "restano verificabili, nessuna shadow execution possibile.", e,
            )

    def _load_shadow_pool(self) -> None:
        """Estrae e quantizza (INT4 simulato) i pesi di shadow_pool_size
        expert, da TUTTI i layer del modello caricato.

        Selezione expert: placeholder round-robin (range(shadow_pool_size)),
        non guidato da carico reale — quali expert cachare in base
        all'hotness è integrazione EAT/Tier Manager (M1/M2), non disponibile
        qui. Questo metodo implementa l'estrazione/quantizzazione/esecuzione
        reale, non la policy di scelta di QUALI expert.

        Layout pesi verificato su FusedMoE reale (2026-08-09, vedi
        _ShadowExpertINT4). Quantizzazione: simmetrica per-tensore, valori
        arrotondati al range INT4 [-8,7] e salvati in int8 — NON bit-packed
        a 4 bit reali. Dimostra la correttezza numerica del round-trip
        quantizza/dequantizza e della matematica SwiGLU sui pesi shadow, non
        il risparmio di memoria reale di un kernel INT4 packed (che
        dimezzerebbe ulteriormente lo storage rispetto a int8) — quello
        resta integrazione kernel separata, fuori scope qui.
        """
        model = self._base.model_runner.model
        layers = model.model.layers
        n_experts = layers[0].block_sparse_moe.experts.num_experts
        expert_ids = list(range(min(self.guard.shadow_pool_size, n_experts)))

        for expert_id in expert_ids:
            per_layer_w13 = []
            per_layer_w2 = []
            for layer in layers:
                experts_module = layer.block_sparse_moe.experts
                w13 = experts_module.w13_weight.data[expert_id]   # (2*intermediate, hidden)
                w2 = experts_module.w2_weight.data[expert_id]     # (hidden, intermediate)
                per_layer_w13.append(_quantize_int4(w13))
                per_layer_w2.append(_quantize_int4(w2))
            self._shadow_pool[expert_id] = _ShadowExpertINT4(per_layer_w13, per_layer_w2)

        log.info(
            "GCSG: shadow pool caricato — %d expert (%s) su %d layer, "
            "quantizzati INT4 (simulato, non packed).",
            len(expert_ids), expert_ids, len(layers),
        )

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

        for row_idx, request_id in enumerate(row_request_ids):
            ctx = GatingContext(
                token_id=row_idx,
                request_id=request_id,
                gating_scores=probs[row_idx].tolist(),
                token_entropy=float(entropy[row_idx]),
            )
            should, _ = self.guard.should_activate_shadow(ctx)
            if should:
                self.guard.run_shadow(
                    ctx, self._shadow_pool,
                    hidden_states=hidden_states[row_idx], layer_id=layer_id,
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
