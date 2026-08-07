"""M3 — Expert Scheduler.

Componenti:
    PT-PEP  : Pre-Tokenization Prompt-Expert Predictor (BERT-small, <3 ms CPU)
    GCSG    : Gating Confidence Shadow Guard
    AER     : Adaptive Expert Replication (stub — replication factor adattivo)

Interfacce verso altri moduli:
    → EAT  (M1): richiesta lookup shard per expert predetto
    → Tier (M2): submit prefetch_queue[], ricezione eviction_candidates[]
    ← vLLM     : hook pre-tokenizzazione (PT-PEP) e post-gating (GCSG)

Vincolo dev: tutto su RTX 3090 24 GB.
    Shadow expert pool (INT4) + expert BF16 attivi devono stare in 24 GB.
    Con Mixtral 8×7B 4-bit: ~14-16 GB model + ~4-6 GB shadow pool.
"""
from .ptpep import PTPEPClassifier, DomainLabel
from .gcsg import GCSGGuard, ShadowExecutionResult
from .aer import AERManager

__all__ = ["PTPEPClassifier", "DomainLabel", "GCSGGuard",
           "ShadowExecutionResult", "AERManager"]
