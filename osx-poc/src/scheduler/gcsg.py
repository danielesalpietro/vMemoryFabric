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
    Shadow pool = top-8 expert INT4 (~3 GB ciascuno ≈ 24 GB totali).
    In pratica: shadow pool ridotto a top-4 per lasciare headroom
    al modello BF16 attivo. Calibrare in Sprint 3.

Hook vLLM: monkey-patch su _run_workers() post-gating, pre-expert-execution.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class GatingContext:
    """Contesto gating per un singolo token/step."""
    token_id:      int
    gating_scores: list[float]     # score per ogni expert (len = n_experts)
    token_entropy: float
    timestamp:     float = field(default_factory=time.monotonic)


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
        theta_contamination: Soglia tasso contaminazione KV-Cache (default 0.05).
    """

    def __init__(
        self,
        theta_gate:          float = 0.85,
        theta_entropy:       float = 0.70,
        theta_contamination: float = 0.05,
    ) -> None:
        self.theta_gate          = theta_gate
        self.theta_entropy       = theta_entropy
        self.theta_contamination = theta_contamination
        self._contamination_counter: int = 0
        self._total_tokens:          int = 0

    # ── core logic ─────────────────────────────────────────────────────────────

    def should_activate_shadow(self, ctx: GatingContext) -> tuple[bool, str]:
        """Decide se attivare shadow execution per un token.

        Returns:
            (should_activate, reason): reason descrive la condizione bloccante
            se should_activate è False.
        """
        raise NotImplementedError("TODO Sprint 3")

    def run_shadow(self, ctx: GatingContext,
                   shadow_pool: dict[int, object]) -> ShadowExecutionResult:
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

    def update_thresholds(self, theta_gate: float | None = None,
                          theta_entropy: float | None = None,
                          theta_contamination: float | None = None) -> None:
        """Aggiorna le soglie a runtime (grid search Sprint 3)."""
        raise NotImplementedError("TODO Sprint 3")

    # ── stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Metriche: activation rate, contamination rate, latenza shadow."""
        raise NotImplementedError("TODO Sprint 3")
