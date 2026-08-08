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
       EngineArgs(worker_cls=GCSGWorker) (parametro confermato presente,
       default "auto").

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

    4. Contaminazione per-request, non globale: verificato che
       ModelInputForGPU (classe base di ModelInputForGPUWithSamplingMetadata,
       il tipo di model_input ricevuto da execute_model()) porta sia
       request_ids_to_seq_ids: Dict[str, List[int]] sia
       seq_group_metadata_list: List[SequenceGroupMetadata] — entrambi
       disponibili per attribuire la contaminazione al request_id giusto
       invece di un contatore unico che mescola sessioni diverse in batch.
       GatingContext.request_id e GCSGGuard.contamination_rate(request_id)
       riflettono questo.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
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

    def run_shadow(self, ctx: GatingContext,
                   shadow_pool: Dict[int, object]) -> ShadowExecutionResult:
        """Esegue shadow execution con expert INT4 dal shadow_pool.

        Sceglie, tra gli expert col gating score più alto, il primo
        effettivamente presente in shadow_pool (il router potrebbe preferire
        un expert non cachato — in quel caso si scende in classifica finché
        non se ne trova uno disponibile, o si desiste).

        Args:
            ctx:         GatingContext del token corrente.
            shadow_pool: Dict expert_id → expert INT4 caricato su VRAM.
                         Ogni valore deve essere callable (forward INT4 reale
                         a integrazione avvenuta — qui invocato senza
                         assumere altro sulla sua interfaccia).

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

        shadow_pool[shadow_expert_id](ctx)   # forward reale wired a integrazione vLLM

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


class GCSGWorker:   # pragma: no cover — richiede vLLM engine live, non unit-testabile
    """Worker vLLM con GCSG cablato — hook verificato, non eseguibile senza GPU.

    Sottoclasse reale di vllm.worker.worker.Worker, istanziata da vLLM stesso
    via EngineArgs(worker_cls=GCSGWorker) quando si costruisce un LLMEngine.
    Il codice sotto segue esattamente la sequenza verificata nel docstring di
    modulo (init_device -> load_model -> shadow pool -> hook su .gate), ma
    NON è coperto da unit test: richiede un LLMEngine reale con un checkpoint
    Mixtral caricato, out of scope per questa sessione (nessun download di
    ~15+ GB di pesi fatto qui). Verificato: import statici, ordine dei
    metodi, e la corrispondenza coi nomi reali delle classi vLLM 0.6.6.post1.
    Da eseguire per la prima volta in un vero smoke test end-to-end (Sprint 3,
    prossima sessione con budget per il download del checkpoint).

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

    def __getattr__(self, name):
        # Delega tutto ciò che non sovrascriviamo esplicitamente al Worker reale
        return getattr(self._base, name)

    def init_device(self) -> None:
        self._base.init_device()

    def load_model(self) -> None:
        """Carica il modello principale, POI lo shadow pool — mai prima.

        Ordine verificato in vllm.executor.gpu_executor.GPUExecutor:
        init_device() gira prima di load_model() e già misura la VRAM libera
        pre-modello — se lo shadow pool venisse caricato in init_device()
        (o prima di super().load_model() qui), il preflight VRAM di
        GCSGGuard vedrebbe VRAM libera artificiosamente alta.
        """
        self._base.load_model()
        self._load_shadow_pool()
        self._register_gate_hooks()

    def _load_shadow_pool(self) -> None:
        """Carica shadow_pool_size expert INT4 — TODO integrazione reale.

        Quali expert cachare (i più "caldi" secondo EAT/Tier Manager) e come
        quantizzarli a INT4 è integrazione M1/M2+M3, deferred: qui c'è
        l'aggancio nel punto giusto del lifecycle, non l'implementazione
        della quantizzazione stessa.
        """
        raise NotImplementedError(
            "TODO Sprint 3 (prossima sessione) — richiede integrazione EAT/Tier "
            "Manager per scegliere quali expert cachare, e un checkpoint Mixtral "
            "reale per quantizzarli a INT4. Il punto di aggancio (dopo "
            "super().load_model(), qui) è verificato e corretto."
        )

    def _register_gate_hooks(self) -> None:
        """Forward hook su ogni layer_i.block_sparse_moe.gate — cattura router_logits.

        Verificato: MixtralMoE.forward() calcola router_logits internamente
        ma non lo restituisce; self.gate (ReplicatedLinear) sì, nel suo
        stesso output. Un hook sul blocco MoE intero vedrebbe solo l'hidden
        state finale già ricombinato dagli expert, non i gating score.
        """
        model = self._base.model_runner.model
        for layer in model.model.layers:
            gate = layer.block_sparse_moe.gate

            def _capture_router_logits(module, inputs, output, _guard=self.guard):
                router_logits, _bias = output
                # TODO Sprint 3: da router_logits + SequenceGroupMetadata del
                # batch corrente a GatingContext per-token/per-request, poi
                # _guard.should_activate_shadow(ctx) — richiede il mapping
                # token-position -> request_id che execute_model() riceve via
                # model_input.request_ids_to_seq_ids (verificato presente).
                return output

            handle = gate.register_forward_hook(_capture_router_logits)
            self._gate_hook_handles.append(handle)

    def execute_model(self, *args, **kwargs):
        """Delega a Worker.execute_model — i gating score arrivano dagli hook
        su .gate (_register_gate_hooks), non ispezionando l'output qui.

        La contaminazione va attribuita per request_id: model_input (primo
        arg posizionale o kwarg, a seconda di come vLLM invoca execute_model)
        porta request_ids_to_seq_ids — verificato presente su
        ModelInputForGPU. Il mapping token/step -> request_id concreto è
        TODO insieme a _load_shadow_pool (stessa integrazione mancante: un
        LLMEngine live).
        """
        return self._base.execute_model(*args, **kwargs)
