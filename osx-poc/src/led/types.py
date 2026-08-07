"""Tipi fondamentali per M4 — RecursiveMAS LED Bridge.

Vedi OSX Design Note v0.3 §2.1 (RecursiveLink), §4.1 (Latent Expert Domain),
§4.2 (LCEPR), §4.3 (RecursiveLink object class).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import time

ExpertID = int
DeviceID = int
LEDID    = int


@dataclass
class RecursiveLinkConfig:
    """Configurazione RecursiveLink — Design Note v0.3 §2.1.

        RecursiveLink(h) = W3·h + W2·σ(W1·h)

    Dimensioni target (Plan v1.0 §2.2, M4): d_h=4096 (Mixtral), hidden=1024,
    ~48M parametri, ~12 MB per coppia di expert in BF16.
    """
    d_hidden_in:  int = 4096
    d_hidden_out: int = 4096
    d_bottleneck: int = 1024
    dtype:        str = "bfloat16"


@dataclass
class LEDConfig:
    """Vincoli di placement per un Latent Expert Domain — LCEPR (Design Note §4.2).

    Dev (single GPU RTX 3090): LCEPR-1 (co-location mandate) è trivialmente
    soddisfatta — un solo device disponibile. Il LED minimo della PoC
    ("LED-2": 2 expert, uno su 3090 uno su 5080 — Plan v1.0 §3.4) è deferred
    all'arrivo della seconda GPU, stesso pattern di AERManager (dual-GPU).
    """
    max_size:   int = 4                 # N esperti per LED — vedi O(N²) scaling, Design Note §5.1
    device_ids: Tuple[DeviceID, ...] = (0,)


@dataclass
class LatentExpertDomain:
    """Un LED: insieme di expert che comunicano via hidden state anziché testo.

    Vincolo LCEPR-1: ogni expert del LED deve avere una replica co-locata
    nello stesso dominio NVLink (dev: stesso device_id).
    """
    led_id:     LEDID
    expert_ids: List[ExpertID]
    device_map: Dict[ExpertID, DeviceID]
    active:     bool = True
    created_ts: float = field(default_factory=time.monotonic)


@dataclass
class HiddenStateTransfer:
    """Risultato di un trasferimento hidden-state tra due expert di un LED.

    Analogo a ShadowExecutionResult (M3, scheduler/gcsg.py) per coerenza col
    Metrics Daemon (latency_ns, hit_rate, contamination_level — Plan §2.2 Fig. 2).
    """
    src_expert_id: ExpertID
    dst_expert_id: ExpertID
    size_bytes:    int
    latency_ms:    float
    projected:     bool = True   # False se fallback su transfer testuale
