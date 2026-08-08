"""M3 — AER: Adaptive Expert Replication.

Gestisce il replication_factor adattivo degli expert in base al carico.
Base weights immutabili. LoRA Delta sync via PCIe (NVLink deferred — no 5080).

In dev (single GPU): AER è stub — nessuna replica fisica possibile, ma NON
silenzioso. Il trigger logic (quando un expert supererebbe la soglia di
carico per meritare una replica) è implementato e testabile a sé:
evaluate_load() valuta la condizione e logga ogni decisione come
WOULD_REPLICATE col motivo, anche se replication_factor() resta sempre 1.
Serve per il soak test/ablation — dimostra che il trigger logic è corretto
prima ancora che arrivi l'hardware (RTX 5080) che la replica la esegue per
davvero.

L'interfaccia è definita ora per garantire compatibilità futura.
"""
from __future__ import annotations
import logging
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)


class AERManager:
    """Adaptive Expert Replication Manager — stub dev, trigger logic reale.

    Args:
        device_ids:         GPU su cui AER opera (dev: solo [0]).
        load_threshold_qps: Soglia di richieste/sec oltre cui un expert
                             "meriterebbe" una replica (default 50.0).

    NOTE: implementazione fisica della replica deferred a dual-GPU setup
    (RTX 5080). In single-GPU, replication_factor è sempre 1 — ma
    evaluate_load() logga comunque ogni WOULD_REPLICATE, col motivo.
    """

    def __init__(
        self,
        device_ids: Optional[list[int]] = None,
        load_threshold_qps: float = 50.0,
    ) -> None:
        self._device_ids = device_ids or [0]  # dev: solo device 0
        self.trigger_conditions = {"load_threshold_qps": load_threshold_qps}
        self._would_replicate_log: List[Tuple[int, float]] = []

    def replication_factor(self, expert_id: int) -> int:
        """Restituisce il replication factor corrente per un expert.

        Dev: sempre 1 (single GPU, nessuna replica fisica possibile) —
        anche per expert che hanno già superato trigger_conditions via
        evaluate_load().
        """
        return 1   # stub — no replication in single-GPU dev

    def evaluate_load(self, expert_id: int, requests_per_second: float) -> bool:
        """Valuta se il carico di un expert supererebbe trigger_conditions.

        Dev: non attiva mai una replica fisica (nessun secondo device su cui
        metterla), ma logga ogni decisione come WOULD_REPLICATE col motivo —
        così il trigger logic è verificabile (soak test/ablation) senza
        aspettare l'hardware che eseguirebbe la replica per davvero.

        Args:
            expert_id:          Expert la cui condizione di carico si valuta.
            requests_per_second: Carico misurato/simulato per quell'expert.

        Returns:
            True se le condizioni per la replica sarebbero soddisfatte.
        """
        threshold = self.trigger_conditions["load_threshold_qps"]
        would_trigger = requests_per_second > threshold
        if would_trigger:
            self._would_replicate_log.append((expert_id, requests_per_second))
            log.info(
                "AER WOULD_REPLICATE expert_id=%d qps=%.1f > threshold=%.1f qps "
                "— no-op: single-GPU dev, nessun secondo device su cui replicare",
                expert_id, requests_per_second, threshold,
            )
        return would_trigger

    def sync_lora_delta(self, expert_id: int, delta: object) -> None:
        """Sincronizza LoRA Delta verso repliche.

        Dev: no-op (nessuna replica).
        Prod: PCIe transfer verso GPU replica (NVLink deferred).
        """
        pass   # stub

    def stats(self) -> dict:
        would_replicate_experts = sorted({expert_id for expert_id, _ in self._would_replicate_log})
        return {
            "replication_enabled": False,
            "reason": "single-GPU dev mode",
            "trigger_conditions": dict(self.trigger_conditions),
            "would_replicate_count": len(self._would_replicate_log),
            "would_replicate_experts": would_replicate_experts,
        }
