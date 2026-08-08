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
        ordered = sorted(candidates, key=lambda e: e.last_access_ts)
        return [
            EvictionCandidate(entry=e, score=e.last_access_ts, policy_used="LRU")
            for e in ordered[:n]
        ]


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
        now = now if now is not None else time.monotonic()
        age = max(0.0, now - entry.last_access_ts)
        # Normalizzati a (0, 1] così alpha/beta pesano contributi comparabili
        # a prescindere dalla scala assoluta di access_count/age — non c'è
        # una scala "naturale" comune tra un contatore e un tempo in secondi.
        recency_component = 1.0 / (1.0 + age)
        freq_component = 1.0 - 1.0 / (1.0 + entry.access_count)

        if context_vec is None:
            weight_total = self.alpha + self.beta
            eff_alpha = self.alpha / weight_total
            eff_beta = self.beta / weight_total
            return eff_alpha * freq_component + eff_beta * recency_component

        # context_vec fornito ma σ è ancora uno stub (PT-PEP arriva in M3):
        # i pesi NON vengono ridistribuiti — gamma resta "sprecato" su un
        # contributo nullo invece di far finta che σ stia facendo qualcosa.
        sigma = 0.0
        return (self.alpha * freq_component
                + self.beta * recency_component
                + self.gamma * sigma)

    def rank(self, candidates: List[EATEntry], n: int,
             context_vec: Optional[list[float]] = None) -> List[EvictionCandidate]:
        """Ordina i candidati per SEE score (più bassi = evict first).

        Fallback su LRU se context_vec è None e gamma > 0.
        """
        now = time.monotonic()
        scored = [
            EvictionCandidate(
                entry=e,
                score=self.score(e, context_vec=context_vec, now=now),
                policy_used="SEE",
            )
            for e in candidates
        ]
        scored.sort(key=lambda c: c.score)
        return scored[:n]
