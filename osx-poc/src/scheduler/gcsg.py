"""M3 — GCSG: Gating Confidence Shadow Guard.

Intercetta i gating scores post-router MoE e attiva una shadow execution
con expert INT4 quando la confidenza è alta e la contaminazione è bassa.

Parametri default (calibrabili a runtime):
    θ_gate          = 0.85   (gating score minimo per shadow)
    θ_entropy       = 0.70   (token entropy massima per shadow)
    θ_contamination = 0.05   (tasso contaminazione KV-Cache massimo)

Shadow execution attivata solo se TUTTE le condizioni:
    - BF16 non disponibile (o budget VRAM insufficiente)
    - gating_score  > θ_gate
    - token_entropy < θ_entropy
    - contamination < θ_contamination

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

Hook vLLM: NON un monkey-patch su _run_workers(). In vLLM 0.6.x ModelRunner
gira in worker process separati (multiprocessing/Ray) — un patch sul processo
principale non si propaga ai worker. L'hook corretto è una sottoclasse
GCSGWorker(Worker) che fa override di execute_model(), passata a LLMEngine via
EngineArgs(worker_cls=GCSGWorker). Punto di aggancio: ModelRunner.execute_model()
(gating scores post-router disponibili lì), non _run_workers() su LLMEngine.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import logging
import time

import pynvml

log = logging.getLogger(__name__)


@dataclass
class GatingContext:
    """Contesto gating per un singolo token/step."""
    token_id:      int
    gating_scores: List[float]     # score per ogni expert (len = n_experts)
    token_entropy: float
    timestamp:     float = field(default_factory=time.monotonic)


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
        theta_contamination: Soglia tasso contaminazione KV-Cache (default 0.05).
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
        self._contamination_counter: int = 0
        self._total_tokens:          int = 0

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

        Returns:
            (should_activate, reason): reason descrive la condizione bloccante
            se should_activate è False.
        """
        raise NotImplementedError("TODO Sprint 3")

    def run_shadow(self, ctx: GatingContext,
                   shadow_pool: Dict[int, object]) -> ShadowExecutionResult:
        """Esegue shadow execution con expert INT4 dal shadow_pool.

        Args:
            ctx:         GatingContext del token corrente.
            shadow_pool: Dict expert_id → expert INT4 caricato su VRAM.

        Returns:
            ShadowExecutionResult con flag contaminazione e latenza.
        """
        raise NotImplementedError("TODO Sprint 3")

    # ── KV-Cache contamination tracking ───────────────────────────────────────

    def contamination_rate(self) -> float:
        """Tasso contaminazione KV-Cache nella finestra attiva (0.0–1.0)."""
        raise NotImplementedError("TODO Sprint 3")

    def reset_contamination_counter(self) -> None:
        """Reset al cambio sessione."""
        raise NotImplementedError("TODO Sprint 3")

    # ── calibration ────────────────────────────────────────────────────────────

    def update_thresholds(self, theta_gate: Optional[float] = None,
                          theta_entropy: Optional[float] = None,
                          theta_contamination: Optional[float] = None) -> None:
        """Aggiorna le soglie a runtime (grid search Sprint 3)."""
        raise NotImplementedError("TODO Sprint 3")

    # ── stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Metriche: activation rate, contamination rate, latenza shadow."""
        raise NotImplementedError("TODO Sprint 3")
