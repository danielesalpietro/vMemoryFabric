#!/usr/bin/env bash
# docker stats sampler, Linux bare-metal equivalent of the docker_stats.csv
# capture used alongside host_sampler.ps1/pcie_sampler.ps1 in
# logs/z8_local/telemetry_fase6a/ (no dedicated .ps1 source for that one —
# it was run ad hoc; this gives it a reusable script).
set -euo pipefail

OUT_DIR="${1:-$HOME/telemetry_new_z8}"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/docker_stats.csv"
DURATION_S="${DURATION_S:-7200}"
INTERVAL_S="${INTERVAL_S:-10}"

echo "timestamp_iso,container,cpu_pct,mem_usage,mem_pct,net_io,block_io,pids" > "$OUT"

deadline=$(( $(date +%s) + DURATION_S ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  ts=$(date -Iseconds)
  rows=$(docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}},{{.NetIO}},{{.BlockIO}},{{.PIDs}}' 2>/dev/null || true)
  if [ -n "$rows" ]; then
    while IFS= read -r row; do
      echo "$ts,$row" >> "$OUT"
    done <<< "$rows"
  fi
  sleep "$INTERVAL_S"
done
