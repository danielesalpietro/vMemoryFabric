#!/usr/bin/env bash
# PCIe RX/TX throughput sampler, Linux bare-metal equivalent of
# logs/z8_local/telemetry_fase6a/pcie_sampler.ps1. Same CSV schema.
set -euo pipefail

OUT_DIR="${1:-$HOME/telemetry_new_z8}"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/pcie_throughput.csv"
DURATION_S="${DURATION_S:-7200}"
INTERVAL_S="${INTERVAL_S:-5}"

echo "timestamp_iso,rxpci_mb_s,txpci_mb_s" > "$OUT"

deadline=$(( $(date +%s) + DURATION_S ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  ts=$(date -Iseconds)
  # nvidia-smi dmon -s t columns: gpu rxpci txpci ... (comment lines start with '#')
  line=$(nvidia-smi dmon -s t -c 1 2>/dev/null | grep -v '^#' | tail -n1 || true)
  if [ -n "$line" ]; then
    read -r _ rx tx _ <<< "$line"
    if [ -n "${rx:-}" ] && [ -n "${tx:-}" ]; then
      echo "$ts,$rx,$tx" >> "$OUT"
    fi
  fi
  sleep "$INTERVAL_S"
done
