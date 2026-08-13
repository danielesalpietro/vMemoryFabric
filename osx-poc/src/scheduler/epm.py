"""M3 — EPM: Expert Position Memory (issue #27).

I/O layer per il checkpoint di hotness EAT (ExpertAccessTable.export_
snapshot()/load_snapshot(), src/eat/eat.py) e per lo storico delle
posizioni dello shadow pool per run — GCSGWorker (scheduler.gcsg) chiama
queste funzioni, non le implementa da sé: la decisione di file system
(dove scrivere, quanto tenere) resta qui, separata dalla logica EAT/GCSG
pura, per lo stesso motivo per cui EAT.load_snapshot() non fa I/O — vedi
la docstring EPM in eat.eat.

Default "attivo, disattivabile" (2026-08-13, richiesta esplicita utente
su issue #27): i default qui sotto sono pensati per essere usati SENZA
argomenti da chi vuole EPM acceso — lo spegnimento è una scelta esplicita
del chiamante (uno script che non passa uno snapshot a
GCSGWorker.configure_eat_snapshot(), o non chiama finalize_epm_run()),
non qualcosa che questo modulo decide da sé.

Storico limitato (2026-08-13, richiesta esplicita utente): un log che
cresce senza limite è uno stesso tipo di rischio già evitato altrove in
questo progetto (vedi captured_router_logits in GCSGWorker, issue #10/#16)
— MAX_HISTORY_RUNS tiene solo gli ultimi N run, FIFO.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

# Path relativi alla working directory di osx-poc — stessa convenzione di
# scheduler.ptpep.model_path in configs/osx_default.yaml. "state/" non è
# codice: vedi .gitignore.
DEFAULT_SNAPSHOT_PATH = Path("state/epm_eat_snapshot.json")
DEFAULT_HISTORY_PATH  = Path("state/epm_run_history.json")

MAX_HISTORY_RUNS = 256


def new_run_id() -> str:
    """ID leggibile e ordinabile per timestamp: <epoch>-<8 hex>."""
    return f"{int(time.time())}-{uuid.uuid4().hex[:8]}"


# ── snapshot (hotness EAT) ───────────────────────────────────────────────────

def load_snapshot_file(path: Path = DEFAULT_SNAPSHOT_PATH) -> Optional[dict]:
    """Legge un dict da EAT.export_snapshot() da file.

    Tollerante per costruzione: file assente, illeggibile o JSON
    corrotto restituiscono None (cold start) invece di sollevare — lo
    stesso principio difensivo di EAT.load_snapshot() lato chiamante
    (GCSGWorker._seed_eat_entries già assorbe un load_snapshot() che
    fallisce; un file mancante non deve arrivare nemmeno a quel punto).
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("EPM: snapshot %s illeggibile (%s) — cold start.", path, e)
        return None


def write_snapshot_file(snapshot: dict, path: Path = DEFAULT_SNAPSHOT_PATH) -> None:
    """Scrive un dict da EAT.export_snapshot() su file, atomicamente
    (write su file temporaneo + rename) — evita un file a metà scritto se
    il processo muore durante il salvataggio."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(snapshot, f)
    tmp.replace(path)


# ── storico run (posizioni iniziali/finali dello shadow pool) ───────────────

def load_history(path: Path = DEFAULT_HISTORY_PATH) -> List[dict]:
    """Lista di record run, più vecchio prima. Tollerante come
    load_snapshot_file(): file assente/corrotto -> lista vuota."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        with path.open("r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("EPM: storico %s illeggibile (%s) — riparte vuoto.", path, e)
        return []
    return data if isinstance(data, list) else []


def positions_match(prev_final: Optional[List[int]], next_initial: Optional[List[int]]) -> bool:
    """True se la posizione finale di un run coincide con quella iniziale
    del run successivo — l'evidenza empirica che il prior caricato ha
    davvero determinato la selezione, non il round-robin di un cold
    start. Confronto insiemistico (sorted), non sull'ordine: l'ordine di
    _select_shadow_expert_ids() a parità di punteggio riflette
    range(n_experts) (vedi la sua docstring), non un segnale di
    posizione."""
    if prev_final is None or next_initial is None:
        return False
    return sorted(prev_final) == sorted(next_initial)


def append_run_record(
    record: dict,
    path: Path = DEFAULT_HISTORY_PATH,
    max_runs: int = MAX_HISTORY_RUNS,
) -> None:
    """Accoda un record run allo storico, troncando al più vecchio se si
    supera max_runs (FIFO — vedi nota di modulo su perché non illimitato).
    Scrittura atomica come write_snapshot_file()."""
    history = load_history(path)
    history.append(record)
    if len(history) > max_runs:
        history = history[-max_runs:]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(history, f, indent=2)
    tmp.replace(path)
