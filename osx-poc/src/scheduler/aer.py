"""M3 — AER: Adaptive Expert Replication.

Gestisce il replication_factor adattivo degli expert in base al carico.
Base weights immutabili. LoRA Delta sync via PCIe (NVLink deferred — no 5080).

In dev (single GPU): AER è stub — nessuna replica fisica possibile.
Sarà attivato con l'arrivo della RTX 5080 (dual-GPU setup).

L'interfaccia è definita ora per garantire compatibilità futura.
"""
from __future__ import annotations


class AERManager:
    """Adaptive Expert Replication Manager — stub dev.

    NOTE: implementazione completa deferred a dual-GPU setup (RTX 5080).
    In single-GPU, replication_factor è sempre 1.
    """

    def __init__(self, device_ids: list[int] | None = None) -> None:
        self._device_ids = device_ids or [0]  # dev: solo device 0

    def replication_factor(self, expert_id: int) -> int:
        """Restituisce il replication factor corrente per un expert.

        Dev: sempre 1 (single GPU, nessuna replica).
        """
        return 1   # stub — no replication in single-GPU dev

    def sync_lora_delta(self, expert_id: int, delta: object) -> None:
        """Sincronizza LoRA Delta verso repliche.

        Dev: no-op (nessuna replica).
        Prod: PCIe transfer verso GPU replica (NVLink deferred).
        """
        pass   # stub

    def stats(self) -> dict:
        return {"replication_enabled": False, "reason": "single-GPU dev mode"}
