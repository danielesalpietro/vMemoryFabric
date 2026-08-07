"""M4 — RecursiveLink: modulo di proiezione hidden-state tra expert.

    RecursiveLink(h) = W3·h + W2·σ(W1·h)     [Design Note v0.3 §2.1, Outer RecursiveLink]

Sostituisce il decode/encode testuale tra expert (~50-200 ms) con trasferimento
hidden-state diretto (~1 ms transfer + ~5 ms projection — Plan v1.0 §3.4).

Training: 100-200 step su MATH500 subset, loss MSE tra hidden state trasmesso
e hidden state ricevuto ottimale, optimizer AdamW lr=1e-4 (Plan v1.0 §2.2, §3.4).

Dimensione target: ~48M parametri, ~12 MB per coppia di expert in BF16
(d_h=4096, hidden=1024).

NOTE dev: nessuna dipendenza torch a livello di modulo — import lazy previsto
dentro init_weights()/forward() per mantenere il package importabile senza
GPU/CUDA (stesso pattern di scheduler/ptpep.py verso onnxruntime).
"""
from __future__ import annotations
from typing import Any, Optional

from .types import RecursiveLinkConfig


class RecursiveLink:
    """Proiezione W3·h + W2·σ(W1·h) tra hidden state di due expert eterogenei.

    Args:
        config: Dimensioni e dtype — vedi RecursiveLinkConfig.
    """

    def __init__(self, config: Optional[RecursiveLinkConfig] = None) -> None:
        self._config  = config or RecursiveLinkConfig()
        self._weights = None   # torch.nn.Module (W1, W2, W3) — allocato in init_weights()

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def init_weights(self, seed: Optional[int] = None) -> None:
        """Inizializza W1, W2, W3 (torch.nn.Linear). Richiede torch."""
        raise NotImplementedError("TODO Sprint 4")

    def load(self, path: str) -> None:
        """Carica pesi RecursiveLink da checkpoint (AER Delta sync — Design Note §4.3)."""
        raise NotImplementedError("TODO Sprint 4")

    def save(self, path: str) -> None:
        """Salva pesi RecursiveLink su checkpoint."""
        raise NotImplementedError("TODO Sprint 4")

    # ── forward / training ────────────────────────────────────────────────────

    def forward(self, hidden_state: Any) -> Any:
        """Applica RecursiveLink(h) = W3·h + W2·σ(W1·h).

        Args:
            hidden_state: torch.Tensor [seq_len, d_hidden_in].

        Returns:
            torch.Tensor [seq_len, d_hidden_out] proiettato per l'expert destinazione.
        """
        raise NotImplementedError("TODO Sprint 4")

    def train_step(self, hidden_state: Any, target_hidden_state: Any) -> float:
        """Un passo di training (AdamW lr=1e-4, loss MSE — Plan v1.0 §3.4).

        Returns:
            Loss value (float).
        """
        raise NotImplementedError("TODO Sprint 4")

    # ── stats ──────────────────────────────────────────────────────────────────

    @property
    def size_bytes(self) -> int:
        """Dimensione pesi in memoria — target ~12 MB per coppia (BF16)."""
        raise NotImplementedError("TODO Sprint 4")

    def stats(self) -> dict:
        """Metriche: loss corrente, numero step training, size_bytes."""
        raise NotImplementedError("TODO Sprint 4")
