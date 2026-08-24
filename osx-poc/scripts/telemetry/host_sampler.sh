#!/usr/bin/env bash
# Host telemetry sampler (CPU/RAM/GPU), Linux bare-metal equivalent of
# logs/z8_local/telemetry_fase6a/host_sampler.ps1 (PowerShell, old
# Docker-on-Windows Z8). Same CSV schema, for continuity with past logs.
#
# ram_free_gb uses /proc/meminfo's MemAvailable (reclaimable-cache-aware),
# not MemFree — closer in spirit to Windows' FreePhysicalMemory than raw
# MemFree would be, but not a bit-for-bit equivalent metric across OSes.
set -euo pipefail

OUT_DIR="${1:-$HOME/telemetry_new_z8}"
mkdir -p "$OUT_DIR"
HOST_OUT="$OUT_DIR/host_perf.csv"
GPU_OUT="$OUT_DIR/gpu_telemetry.csv"
DURATION_S="${DURATION_S:-7200}"
INTERVAL_S="${INTERVAL_S:-10}"

echo "timestamp_iso,cpu_pct,ram_used_gb,ram_free_gb,ram_total_gb" > "$HOST_OUT"
echo "timestamp_iso,name,util_gpu_pct,util_mem_pct,mem_used_mib,mem_total_mib,power_w,temp_c" > "$GPU_OUT"

cpu_pct() {
  read -r _ u1 n1 s1 i1 w1 irq1 sirq1 st1 _ < /proc/stat
  sleep 0.3
  read -r _ u2 n2 s2 i2 w2 irq2 sirq2 st2 _ < /proc/stat
  idle1=$((i1 + w1)); idle2=$((i2 + w2))
  total1=$((u1 + n1 + s1 + i1 + w1 + irq1 + sirq1 + st1))
  total2=$((u2 + n2 + s2 + i2 + w2 + irq2 + sirq2 + st2))
  dt=$((total2 - total1)); di=$((idle2 - idle1))
  if [ "$dt" -le 0 ]; then
    echo "0.0"
  else
    awk -v dt="$dt" -v di="$di" 'BEGIN{printf "%.1f", (dt-di)*100.0/dt}'
  fi
}

deadline=$(( $(date +%s) + DURATION_S ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  ts=$(date -Iseconds)
  cpu=$(cpu_pct)

  read -r ram_total_kb ram_avail_kb <<EOF_MEM
$(awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{print t, a}' /proc/meminfo)
EOF_MEM
  ram_total_gb=$(awk -v k="$ram_total_kb" 'BEGIN{printf "%.2f", k/1024/1024}')
  ram_free_gb=$(awk -v k="$ram_avail_kb" 'BEGIN{printf "%.2f", k/1024/1024}')
  ram_used_gb=$(awk -v t="$ram_total_gb" -v f="$ram_free_gb" 'BEGIN{printf "%.2f", t-f}')
  echo "$ts,$cpu,$ram_used_gb,$ram_free_gb,$ram_total_gb" >> "$HOST_OUT"

  gpu_line=$(nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || true)
  if [ -n "$gpu_line" ]; then
    echo "$ts,$gpu_line" >> "$GPU_OUT"
  fi

  sleep "$INTERVAL_S"
done
