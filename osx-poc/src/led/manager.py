"""M4 — LED Manager: gestisce ciclo di vita dei Latent Expert Domain e LCEPR.

Interfacce verso altri moduli (Plan v1.0 §2.2, Figura 2):
    ← ES  (M3): led_assignment[], recursivelink_weights
    → EAT (M1): residency check per RecursiveLink object class (pinned EMH-1)
    → AER (M3): LED AER coupling — LCEPR-3, replication factor sincronizzato
    → Metrics: latency_ns, hit_rate, contamination_level

Enforce Latent-Coupled Expert Placement Rule (LCEPR) — Design Note v0.3 §4.2:
    LCEPR-1  Co-location mandate: tutti i membri LED sullo stesso rack/device.
    LCEPR-2  RecursiveLink residency: RecursiveLink pinned sulla stessa GPU
             dell'expert corrispondente — nessuna esecuzione cross-GPU.
    LCEPR-3  LED AER coupling: quando AER replica un expert del LED, tutti i
             co-membri devono adeguare il replication factor simultaneamente.
    LCEPR-4  OCS circuit inter-LED: deferred — richiede topologia multi-rack
             (NVL72), fuori scope della PoC single-node.

Vincolo dev (single RTX 3090): il LED minimo della PoC ("LED-2": 2 expert, uno
su 3090 uno su 5080 — Plan v1.0 §3.4) è deferred all'arrivo della seconda GPU.
Fino ad allora ogni LED è vincolato a device_ids in LEDConfig (default: [0]).
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from .recursivelink import RecursiveLink
from .types import (
    DeviceID,
    ExpertID,
    HiddenStateTransfer,
    LatentExpertDomain,
    LEDConfig,
    LEDID,
)


class LEDManager:
    """Gestisce creazione, validazione e dissoluzione dei LED.

    Args:
        config: Vincoli di placement (LEDConfig) — device disponibili, max_size.
    """

    def __init__(self, config: Optional[LEDConfig] = None) -> None:
        self._config      = config or LEDConfig()
        self._leds:  Dict[LEDID, LatentExpertDomain] = {}
        self._links: Dict[Tuple[ExpertID, ExpertID], RecursiveLink] = {}
        self._next_led_id: LEDID = 0

    # ── LED lifecycle ──────────────────────────────────────────────────────────

    def create_led(self, expert_ids: List[ExpertID],
                   device_map: Dict[ExpertID, DeviceID]) -> LatentExpertDomain:
        """Crea un nuovo LED e valida LCEPR-1 (co-location mandate).

        Raises:
            ValueError: len(expert_ids) > max_size, device_map incompleta,
                        o un device non è in self._config.device_ids (LCEPR-1).
        """
        raise NotImplementedError("TODO Sprint 4")

    def dissolve_led(self, led_id: LEDID) -> None:
        """Dissolve un LED — RecursiveLink modules evicted da EMH-1 (Design Note §4.3)."""
        raise NotImplementedError("TODO Sprint 4")

    def get_led(self, led_id: LEDID) -> Optional[LatentExpertDomain]:
        """Restituisce il LED se attivo, None altrimenti."""
        raise NotImplementedError("TODO Sprint 4")

    # ── LCEPR enforcement ──────────────────────────────────────────────────────

    def validate_lcepr(self, led: LatentExpertDomain) -> bool:
        """Verifica LCEPR-1 e LCEPR-2 per il LED dato.

        LCEPR-3 (AER coupling) e LCEPR-4 (OCS inter-rack) sono deferred —
        richiedono rispettivamente dual-GPU (AER) e topologia NVL72 multi-rack.
        """
        raise NotImplementedError("TODO Sprint 4")

    def on_expert_migrated(self, expert_id: ExpertID, new_device_id: DeviceID) -> None:
        """Hook AER: ri-valida LCEPR-1 quando un expert migra (LCEPR-3 coupling)."""
        raise NotImplementedError("TODO Sprint 4")

    # ── RecursiveLink registry ────────────────────────────────────────────────

    def get_or_create_link(self, src_expert_id: ExpertID,
                           dst_expert_id: ExpertID) -> RecursiveLink:
        """Restituisce il RecursiveLink per la coppia, creandolo se assente.

        Un RecursiveLink per coppia è condiviso tra tutti i round di ricorsione
        (Design Note §4.3) — O(N) oggetti per catena sequenziale, O(N²) per
        LED full-mesh (vedi Open Problem §5.1).
        """
        raise NotImplementedError("TODO Sprint 4")

    # ── hidden state transfer ─────────────────────────────────────────────────

    def transfer(self, src_expert_id: ExpertID, dst_expert_id: ExpertID,
                hidden_state: Any) -> HiddenStateTransfer:
        """Trasferisce e proietta hidden state tra due expert dello stesso LED.

        Richiede LCEPR-2 soddisfatta (RecursiveLink co-residente con l'expert
        sorgente). Target latenza: ~1 ms transfer + ~5 ms projection (Plan §3.4).
        """
        raise NotImplementedError("TODO Sprint 4")

    # ── stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Metriche: numero LED attivi, RecursiveLink totali, budget EMH-1 (MB)."""
        raise NotImplementedError("TODO Sprint 4")
