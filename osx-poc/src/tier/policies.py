"""M2 — Eviction policies: SEE + LRU fallback.

SEE — Semantic Expert Eviction:
    score(shard) = α·freq + β·recency + γ·σ(shard, context)
    Parametri default: α=0.3, β=0.3, γ=0.4
    σ(shard, context) = similarità semantica shard/contesto corrente (da PT-PEP).

    In Sprint 2: σ è uno stub (γ=0, fallback LRU puro).
    In Sprint 3: σ viene alimentato da PT-PEP (M3) con il vettore semantico.

LRU — fallback se SEE non disponibile o contesto non definito.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import time

from eat.types import EATEntry, ExpertID, ShardID, Tier


@dataclass
class EvictionCandidate:
    entry:       EATEntry
    score:       float    # score SEE o LRU (più basso = candidato migliore)
    policy_used: str      # "SEE" | "LRU"


class LRUPolicy:
    """Eviction LRU pura — fallback sicuro, nessuna dipendenza esterna."""

    def rank(self, candidates: List[EATEntry], n: int) -> List[EvictionCandidate]:
        """Ordina i candidati per last_access_ts crescente (meno recenti prima).

        Args:
            candidates: Lista di EATEntry nel tier target.
            n:          Numero di candidati da restituire.

        Returns:
            Top-n candidati all'eviction ordinati per priorità.
        """
        raise NotImplementedError("TODO Sprint 2")


class SEEPolicy:
    """SEE — Semantic Expert Eviction.

    Args:
        alpha: Peso frequenza accesso         (default 0.3).
        beta:  Peso recency                    (default 0.3).
        gamma: Peso similarità semantica σ     (default 0.4).
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.3, gamma: float = 0.4) -> None:
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "α+β+γ deve essere 1.0"
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self._lru  = LRUPolicy()   # fallback interno

    def score(self, entry: EATEntry,
              context_vec: Optional[list[float]] = None,
              now: Optional[float] = None) -> float:
        """Calcola SEE score per una entry.

        Args:
            entry:       EATEntry da valutare.
            context_vec: Vettore semantico contesto corrente da PT-PEP (None = stub).
            now:         Timestamp corrente (monotonic). None = time.monotonic().

        Returns:
            Score SEE (più alto = mantenere in tier; più basso = evict).

        NOTE: se context_vec è None, gamma viene ignorato e il peso
        viene ridistribuito su alpha e beta (LRU ponderata).
        """
        raise NotImplementedError("TODO Sprint 2 — σ stub; Sprint 3 integra PT-PEP")

    def rank(self, candidates: List[EATEntry], n: int,
             context_vec: Optional[list[float]] = None) -> List[EvictionCandidate]:
        """Ordina i candidati per SEE score (più bassi = evict first).

        Fallback su LRU se context_vec è None e gamma > 0.
        """
        raise NotImplementedError("TODO Sprint 2")
