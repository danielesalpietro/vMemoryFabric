"""M4 — RecursiveMAS LED Bridge.

Componenti:
    RecursiveLink : proiezione W3·h + W2·σ(W1·h) tra hidden state eterogenei
                    (Design Note v0.3 §2.1).
    LEDManager    : ciclo di vita dei Latent Expert Domain + enforcement LCEPR
                    (Design Note v0.3 §4.1-§4.2).

Interfacce verso altri moduli (Plan v1.0 §2.2, Figura 2):
    → EAT  (M1): RecursiveLink gestito come shard subtype, pinned in EMH-1
    → Tier (M2): nessuna eviction per RecursiveLink attivi (residency policy)
    ← ES   (M3): led_assignment[], recursivelink_weights
    → AER  (M3): LED AER coupling — LCEPR-3

Vincolo dev: il LED minimo della PoC ("LED-2", Plan v1.0 §3.4) richiede 2 GPU
(un expert per device). Su singola RTX 3090 il LED è a device singolo —
deferred all'arrivo della RTX 5080, stesso pattern di AERManager
(src/scheduler/aer.py).
"""
from .recursivelink import RecursiveLink
from .manager import LEDManager
from .types import (
    LEDConfig,
    LatentExpertDomain,
    RecursiveLinkConfig,
    HiddenStateTransfer,
)

__all__ = [
    "RecursiveLink",
    "LEDManager",
    "LEDConfig",
    "LatentExpertDomain",
    "RecursiveLinkConfig",
    "HiddenStateTransfer",
]
