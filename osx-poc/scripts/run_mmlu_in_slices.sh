#!/usr/bin/env bash
# Orchestratore issue #10/#16 (2026-08-10): processa i 570 prompt MMLU a
# fette, ognuna come processo Docker SEPARATO (GCSGWorker fresco ogni
# volta) — l'unico pattern verificato funzionare in modo affidabile finora.
# Ogni chiamata generate() singola dentro un processo fresco (n=8/16/32
# isolati) ha sempre completato pulita; qualunque riuso dello stesso
# processo per una seconda chiamata (n=64 singola, n=64 a blocchi
# in-process) si e' bloccato in modo riproducibile. Non root-causato — vedi
# LOGBOOK 2026-08-10 — questo aggira il problema invece di risolverlo.
#
# Ogni fetta scrive UNA riga JSON su --results-file (append, condiviso tra
# tutte le fette) — se una fetta si blocca, il file mostra esattamente fin
# dove si e' arrivato, senza perdere le fette gia' completate.
#
# Usage:
#   scripts/run_mmlu_in_slices.sh [slice_size] [total] [results_file]
set -uo pipefail   # NON -e: una fetta che fallisce non deve fermare le altre

SLICE_SIZE="${1:-32}"
TOTAL="${2:-570}"
RESULTS_FILE="${3:-/workspace/osx-poc/mmlu_results_$(date +%Y%m%d_%H%M%S).jsonl}"

echo "=== Orchestratore MMLU a fette — slice_size=$SLICE_SIZE, total=$TOTAL, results_file=$RESULTS_FILE ==="

start=0
slice_index=0
while [ "$start" -lt "$TOTAL" ]; do
    end=$((start + SLICE_SIZE))
    if [ "$end" -gt "$TOTAL" ]; then
        end=$TOTAL
    fi
    n=$((end - start))
    echo ""
    echo "--- Fetta $slice_index: [$start:$end) ($n prompt), processo Docker separato ---"

    # Timeout alzati 2026-08-11 (vedi LOGBOOK): una fetta che completa in
    # 850s+ e' normale sotto WSL2/cpu_offload_gb, non un segno di stallo —
    # i vecchi 300s/250s uccidevano fette che stavano solo per finire.
    PYTHONPATH=src timeout 3000 python scripts/eval_mmlu_gcsg.py \
        --prompt-start "$start" \
        --max-prompts "$n" \
        --results-file "$RESULTS_FILE" \
        --watchdog-timeout 2700

    status=$?
    if [ "$status" -ne 0 ]; then
        echo "--- Fetta $slice_index [$start:$end) FALLITA (exit $status) — vedi $RESULTS_FILE per l'ultima fetta completata con successo. Proseguo con la prossima. ---"
    else
        echo "--- Fetta $slice_index [$start:$end) OK ---"
    fi

    start=$end
    slice_index=$((slice_index + 1))
done

echo ""
echo "=== Orchestratore completato — $slice_index fette tentate. Risultati in $RESULTS_FILE ==="
