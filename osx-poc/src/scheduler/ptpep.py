"""M3 — PT-PEP: Pre-Tokenization Prompt-Expert Predictor.

Classifica il dominio semantico di un prompt PRIMA della tokenizzazione,
permettendo il prefetch degli expert MoE pertinenti.

Modello: BERT-small fine-tuned su 8 domini.
Runtime: ONNX su onnxruntime CPU (target < 3 ms p99 su Xeon 6244).
Input:   testo grezzo (stringa).
Output:  DomainLabel + confidence scores per tutti i domini.

Pipeline:
    1. PT-PEP predice dominio → expert_ids probabili
    2. Expert IDs → prefetch_queue (EAT + Tier Manager)
    3. Shard rilevanti vengono promossi verso VRAM prima del forward pass

Training (Sprint 3):
    Dataset: ~2.000 prompt da OpenHermes, MetaMathQA, CodeAlpaca,
             PubMedQA, LegalBench + labeling automatico.
    Fine-tuning: BERT-small (HuggingFace) → ONNX export.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class DomainLabel(str, Enum):
    """8 domini semantici per PT-PEP."""
    CODING   = "coding"
    MATH     = "math"
    LANGUAGE = "language"
    SCIENCE  = "science"
    MEDICAL  = "medical"
    LEGAL    = "legal"
    CREATIVE = "creative"
    GENERAL  = "general"


@dataclass
class PTPEPPrediction:
    """Output di una singola inferenza PT-PEP."""
    domain:      DomainLabel
    confidence:  float                     # P(domain | prompt) per il top-1
    all_scores:  Dict[DomainLabel, float]  # distribuzione completa
    latency_ms:  float                     # latenza inferenza effettiva
    expert_ids:  List[int]                 # expert IDs predetti per il dominio


class PTPEPClassifier:
    """Classifica il dominio semantico di un prompt via BERT-small ONNX.

    Args:
        model_path:    Path al file ONNX esportato (None = non caricato/stub).
        expert_map:    Dict dominio → lista expert_ids (da config OSX).
                       NOTE: su Mixtral 8x7B il routing MoE non ha identità
                       semantica fissa per expert — i gating score dipendono
                       dall'input, non da un mapping dominio->expert hardcoded.
                       expert_map è quindi una statistica di routing empirica
                       (derivata da sample set), non ground truth. Dichiararlo
                       esplicitamente in config con un commento tipo
                       "# empirical routing statistics, not guaranteed".
        confidence_th: Soglia minima per considerare una predizione affidabile.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        expert_map: Optional[Dict[DomainLabel, List[int]]] = None,
        confidence_th: float = 0.6,
    ) -> None:
        self._model_path    = model_path
        self._expert_map    = expert_map or {}
        self._confidence_th = confidence_th
        self._session       = None   # onnxruntime.InferenceSession — caricato in load()

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Carica il modello ONNX. Chiamare una volta all'avvio.

        NOTE: in stub mode (model_path=None), load() è no-op e predict()
        restituisce sempre DomainLabel.GENERAL con confidence=0.0.
        """
        raise NotImplementedError("TODO Sprint 3 — richiede modello ONNX fine-tuned")

    def unload(self) -> None:
        """Rilascia il modello dalla memoria."""
        raise NotImplementedError("TODO Sprint 3")

    # ── inference ──────────────────────────────────────────────────────────────

    def predict(self, prompt: str) -> PTPEPPrediction:
        """Predice dominio e expert IDs per un prompt.

        Args:
            prompt: Testo grezzo (pre-tokenizzazione vLLM).

        Returns:
            PTPEPPrediction con dominio, confidence, expert_ids, latenza.

        NOTE: se confidence < confidence_th, restituisce DomainLabel.GENERAL
        come fallback sicuro (nessun prefetch speculativo).
        """
        raise NotImplementedError("TODO Sprint 3")

    def predict_batch(self, prompts: List[str]) -> List[PTPEPPrediction]:
        """Batch inference per throughput elevato (batch_size configurabile)."""
        raise NotImplementedError("TODO Sprint 3")

    # ── stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Hit rate, latenza P50/P95/P99, distribuzione domini."""
        raise NotImplementedError("TODO Sprint 3")
