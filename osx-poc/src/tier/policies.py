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

import time
from dataclasses import dataclass

from eat.types import EATEntry


@dataclass
class EvictionCandidate:
    entry:       EATEntry
    score:       float    # score SEE o LRU (più basso = candidato migliore)
    policy_used: str      # "SEE" | "LRU"


class LRUPolicy:
    """Eviction LRU pura — fallback sicuro, nessuna dipendenza esterna."""

    def rank(self, candidates: list[EATEntry], n: int) -> list[EvictionCandidate]:
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
              context_vec: list[float] | None = None,
              now: float | None = None) -> float:
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

    def rank(self, candidates: list[EATEntry], n: int,
             context_vec: list[float] | None = None) -> list[EvictionCandidate]:
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

    def classify_hot_cold(
        self, entries: list[EATEntry], hot_fraction: float = 0.5,
        now: float | None = None,
    ) -> tuple[list[int], list[int]]:
        """Issue #33 Fase 0 — separa expert_id "caldi" (candidati VRAM/GPU
        shadow pool) da "freddi" (candidati DDR4-resident/CPU shadow pool —
        issue #33 Fase 2) usando lo score() già esistente, non una formula
        nuova — esattamente quello che Fase 0 del piano chiedeva ("riusare
        SEEPolicy invece di inventarne una nuova").

        Criterio validato contro il codice reale di
        exllamav3/model/moe_cpu_host.py, non per sola analogia: lo stesso
        segnale — quanti token sono stati assegnati a un expert — decide
        lì quali esperti vengono streammati "caldi" verso la GPU e quali
        restano "coda fredda" calcolata su CPU con AVX-512 (vedi
        osx-poc/reports/component_reuse_analysis.md §2.1, verificato sul
        sorgente exllamav3, non sul solo README). access_count di EATEntry
        è esattamente quel segnale — lo stesso già pesato dal termine
        freq_component di score().

        Deliberatamente NON usa context_vec/σ (l'estensione proposta in
        issue #21, collegare PT-PEP a σ): quel collegamento resta un
        miglioramento non ancora misurato, con un rischio di scope
        mismatch dichiarato nella issue stessa (σ pensato per lo shard
        corrente, PT-PEP opera a livello dell'intero prompt) — "misurare
        prima di collegare", non assumere che sia un miglioramento.
        Restare sul path context_vec=None (LRU ponderata alpha/beta) tiene
        questa funzione sul solo segnale già validato, senza ereditare un
        rischio non ancora verificato.

        Aggrega per expert_id sommando lo score di ogni entry (una per
        layer/shard_idx) — stesso principio di aggregazione già usato in
        GCSGWorker._select_shadow_expert_ids() per l'hotness (lì su
        access_count grezzo, qui via score()): un expert con hotness
        sparsa su più layer non deve perdere contro uno concentrato su un
        solo layer, se il totale è maggiore.

        hot_fraction è una frazione parametrizzabile, non una soglia
        hardcoded: il punto di taglio esatto tra "abbastanza caldo per la
        VRAM" e "abbastanza freddo per restare CPU" dipende dal costo
        relativo reale transfer-vs-compute su questo hardware — misura non
        ancora fatta (issue #24, richiede budget POD). Fase 0 fornisce la
        struttura del criterio, non il valore calibrato — la calibrazione
        resta esplicitamente aperta per quando quel dato esisterà.

        Cold start onesto: con tutte le entry ad access_count=0 (nessun
        traffico reale ancora instradato), lo score si riduce al solo
        termine di recency — stesso principio già dichiarato in
        _select_shadow_expert_ids() per il suo caso round-robin
        equivalente, non un comportamento nascosto qui.

        Args:
            entries:      EATEntry del tier candidato (tipicamente
                          tier_manager.eat.get_tier(Tier.DDR4)).
            hot_fraction: Frazione di expert_id distinti (non di entry —
                          un expert_id ha un'entry per layer) da
                          classificare "caldi", in (0, 1]. Default 0.5.
            now:          Timestamp corrente (monotonic). None =
                          time.monotonic().

        Returns:
            (hot_ids, cold_ids): expert_id ordinati per score aggregato
            decrescente, spaccati al punto round(n * hot_fraction)
            (minimo 1 se ci sono entry).
        """
        assert 0.0 < hot_fraction <= 1.0, "hot_fraction deve essere in (0, 1]"
        now = now if now is not None else time.monotonic()

        scores: dict[int, float] = {}
        for entry in entries:
            scores[entry.expert_id] = (
                scores.get(entry.expert_id, 0.0) + self.score(entry, now=now)
            )
        if not scores:
            return [], []

        ranked = sorted(scores, key=lambda expert_id: scores[expert_id], reverse=True)
        hot_count = max(1, round(len(ranked) * hot_fraction))
        return ranked[:hot_count], ranked[hot_count:]
