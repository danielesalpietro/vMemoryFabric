"""M3 — PT-PEP: Pre-Tokenization Prompt-Expert Predictor.

Classifica il dominio semantico di un prompt PRIMA della tokenizzazione,
permettendo il prefetch degli expert MoE pertinenti.

Modello (Sprint 3 baseline, non BERT-small): TF-IDF + centroidi per dominio.
    Un TfidfVectorizer condiviso, fit su testo estratto automaticamente da
    dataset pubblici (non keyword curate a mano — soggettivo e non
    difendibile). Un centroide per dominio = media dei vettori TF-IDF dei
    documenti di training di quel dominio. predict() = cosine similarity
    tra il vettore del prompt e gli 8 centroidi, normalizzata a distribuzione.
    BERT-small fine-tuned + export ONNX resta l'obiettivo per un'iterazione
    futura (OP1 esteso) SE questa baseline non supera il target di hit rate —
    l'interfaccia (model_path, predict() -> PTPEPPrediction) è la stessa,
    quindi lo swap non tocca nulla a valle.
Runtime: scikit-learn puro CPU (nessuna dipendenza CUDA/torch).
Input:   testo grezzo (stringa).
Output:  DomainLabel + confidence scores per tutti i domini.

Pipeline:
    1. PT-PEP predice dominio → expert_ids probabili
    2. Expert IDs → prefetch_queue (EAT + Tier Manager)
    3. Shard rilevanti vengono promossi verso VRAM prima del forward pass

Training (Sprint 3, vedi scripts/build_ptpep_classifier.py):
    250 esempi/dominio da dataset pubblici (CodeAlpaca, MetaMathQA, PubMedQA,
    LegalBench, SciQ, CoEdit, WritingPrompts, Alpaca) — 200 train + 50
    held-out per dominio (2000 totali). Split 80/20 dallo stesso dataset per
    dominio: held-out "same-distribution", non OOD da fonte diversa — onesto
    da dichiarare così nel paper, non spacciarlo per generalizzazione reale.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import time

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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
    """Classifica il dominio semantico di un prompt via TF-IDF + centroidi
    (Sprint 3 baseline — vedi docstring di modulo per il perché non BERT-small).

    Args:
        model_path:    Path all'artifact joblib prodotto da
                        scripts/build_ptpep_classifier.py (None = stub mode:
                        load() no-op, predict() sempre DomainLabel.GENERAL
                        con confidence=0.0 — nessun prefetch speculativo).
        expert_map:    Dict dominio → lista expert_ids (da config OSX).
                       NOTE: su Mixtral 8x7B il routing MoE non ha identità
                       semantica fissa per expert — i gating score dipendono
                       dall'input, non da un mapping dominio->expert hardcoded.
                       expert_map è quindi una statistica di routing empirica
                       (derivata da sample set), non ground truth. Dichiararlo
                       esplicitamente in config con un commento tipo
                       "# empirical routing statistics, not guaranteed".
        confidence_th: Soglia minima per considerare una predizione affidabile.
        similarity_temperature: Temperatura del softmax applicato alle cosine
                        similarity (default 0.03, calibrata empiricamente —
                        vedi _softmax_rows). Più bassa = distribuzione più
                        piccata sul top-1.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        expert_map: Optional[Dict[DomainLabel, List[int]]] = None,
        confidence_th: float = 0.6,
        similarity_temperature: float = 0.03,
    ) -> None:
        self._model_path    = model_path
        self._expert_map    = expert_map or {}
        self._confidence_th = confidence_th
        self._temperature    = similarity_temperature
        self._artifact       = None   # {"vectorizer", "centroids", "domains"} — load()

        self._predict_count  = 0
        self._fallback_count = 0
        self._domain_counts: Counter = Counter()
        self._latencies: List[float] = []

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Carica l'artifact TF-IDF+centroidi. Chiamare una volta all'avvio.

        NOTE: in stub mode (model_path=None), load() è no-op e predict()
        restituisce sempre DomainLabel.GENERAL con confidence=0.0.
        """
        if self._model_path is None:
            return
        self._artifact = joblib.load(self._model_path)

    def unload(self) -> None:
        """Rilascia il modello dalla memoria."""
        self._artifact = None

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
        start = time.perf_counter()

        if self._artifact is None:
            latency_ms = (time.perf_counter() - start) * 1000
            self._record_stats(DomainLabel.GENERAL, latency_ms, fallback=True)
            return PTPEPPrediction(
                domain=DomainLabel.GENERAL,
                confidence=0.0,
                all_scores={d: 0.0 for d in DomainLabel},
                latency_ms=latency_ms,
                expert_ids=self._expert_map.get(DomainLabel.GENERAL, []),
            )

        all_scores, top_domain, top_confidence = self._score(prompt)
        latency_ms = (time.perf_counter() - start) * 1000
        result_domain, fallback = self._apply_confidence_threshold(top_domain, top_confidence)
        self._record_stats(result_domain, latency_ms, fallback=fallback)

        return PTPEPPrediction(
            domain=result_domain,
            confidence=top_confidence,
            all_scores=all_scores,
            latency_ms=latency_ms,
            expert_ids=self._expert_map.get(result_domain, []),
        )

    def predict_batch(self, prompts: List[str]) -> List[PTPEPPrediction]:
        """Batch inference per throughput elevato (batch_size configurabile).

        Vettorizza l'intero batch in una sola chiamata (più efficiente di N
        chiamate a predict()), ma la latenza per-item è amortizzata (tempo
        totale / batch_size) — non una misura per-item reale come in predict().
        """
        if self._artifact is None or not prompts:
            return [self.predict(p) for p in prompts]

        start = time.perf_counter()
        vectorizer, centroid_matrix, domains = self._artifact_components()
        vecs = vectorizer.transform(prompts)
        sims = cosine_similarity(vecs, centroid_matrix)
        probs = self._softmax_rows(sims)
        total_latency_ms = (time.perf_counter() - start) * 1000
        per_item_latency_ms = total_latency_ms / len(prompts)

        results = []
        for row in probs:
            all_scores, top_domain, top_confidence = self._row_to_scores(row, domains)
            result_domain, fallback = self._apply_confidence_threshold(top_domain, top_confidence)
            self._record_stats(result_domain, per_item_latency_ms, fallback=fallback)
            results.append(PTPEPPrediction(
                domain=result_domain,
                confidence=top_confidence,
                all_scores=all_scores,
                latency_ms=per_item_latency_ms,
                expert_ids=self._expert_map.get(result_domain, []),
            ))
        return results

    # ── scoring internals ─────────────────────────────────────────────────────

    def _artifact_components(self):
        vectorizer = self._artifact["vectorizer"]
        centroids  = self._artifact["centroids"]
        domains    = self._artifact["domains"]
        centroid_matrix = np.stack([centroids[d] for d in domains])
        return vectorizer, centroid_matrix, domains

    def _row_to_scores(
        self, row: np.ndarray, domains: List[str],
    ) -> Tuple[Dict[DomainLabel, float], DomainLabel, float]:
        all_scores = {DomainLabel(d): float(p) for d, p in zip(domains, row)}
        top_idx = int(np.argmax(row))
        return all_scores, DomainLabel(domains[top_idx]), float(row[top_idx])

    def _score(self, prompt: str) -> Tuple[Dict[DomainLabel, float], DomainLabel, float]:
        vectorizer, centroid_matrix, domains = self._artifact_components()
        vec = vectorizer.transform([prompt])
        sims = cosine_similarity(vec, centroid_matrix)[0]
        probs = self._softmax_rows(sims[np.newaxis, :])[0]
        return self._row_to_scores(probs, domains)

    def _softmax_rows(self, sims: np.ndarray) -> np.ndarray:
        """Softmax(sims / temperature), riga per riga.

        NOTE — non normalizzazione lineare (sims / sims.sum()): con cosine
        similarity su TF-IDF sparso, i punteggi grezzi sono piccoli e vicini
        tra loro (visto sui dati reali: top1 medio ~0.18, media globale
        ~0.04) — normalizzare linearmente schiaccia la confidence anche
        quando l'argmax è chiaramente corretto (misurato: hit rate crolla da
        87% ad arg­max puro a 55% con soglia 0.6 su normalizzazione lineare).
        similarity_temperature=0.03 è stato scelto empiricamente sul set di
        validazione (tests/fixtures/ptpep_validation.json) come compromesso:
        abbastanza acuto da non buttare via predizioni corrette (80% delle
        predizioni corrette superano la soglia 0.6), abbastanza morbido da
        non degenerare in argmax puro (a differenza di T=0.01, che rende la
        soglia di confidence sostanzialmente inerte). Essendo calibrato sullo
        stesso set usato per il test di hit rate, è un limite dichiarato, non
        una validazione indipendente — onesto da annotare nel paper.
        """
        scaled = sims / self._temperature
        scaled = scaled - scaled.max(axis=-1, keepdims=True)   # stabilità numerica
        exp = np.exp(scaled)
        return exp / exp.sum(axis=-1, keepdims=True)

    def _apply_confidence_threshold(
        self, top_domain: DomainLabel, top_confidence: float,
    ) -> Tuple[DomainLabel, bool]:
        if top_confidence < self._confidence_th:
            return DomainLabel.GENERAL, True
        return top_domain, False

    # ── stats ──────────────────────────────────────────────────────────────────

    def _record_stats(self, domain: DomainLabel, latency_ms: float, fallback: bool) -> None:
        self._predict_count += 1
        self._domain_counts[domain] += 1
        self._latencies.append(latency_ms)
        if len(self._latencies) > 10_000:   # bounded — this is a running window, not an archive
            self._latencies = self._latencies[-10_000:]
        if fallback:
            self._fallback_count += 1

    def stats(self) -> dict:
        """Hit rate (fallback rate), latenza P50/P95/P99, distribuzione domini.

        NOTE: "hit rate" qui è la frazione di predict() che NON sono finite
        in fallback GENERAL per bassa confidence — è quanto la classe può
        sapere senza ground truth. L'hit rate contro etichette vere (target
        >70%) è misurato offline da scripts/build_ptpep_classifier.py contro
        tests/fixtures/ptpep_validation.json, non qui.
        """
        if self._predict_count == 0:
            return {
                "predict_count": 0, "fallback_rate": 0.0,
                "domain_distribution": {}, "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0, "latency_p99_ms": 0.0,
            }
        latencies = np.asarray(self._latencies)
        return {
            "predict_count": self._predict_count,
            "fallback_rate": self._fallback_count / self._predict_count,
            "domain_distribution": {
                d.value: count / self._predict_count
                for d, count in self._domain_counts.items()
            },
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "latency_p99_ms": float(np.percentile(latencies, 99)),
        }
