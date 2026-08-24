# Issue #33 Fase 6a - PCIe RX/TX throughput sampler, timestamped, per
# capire quanto traffico CPU<->GPU genera route_forward() sugli expert
# freddi (hidden_states.cpu() ad ogni dispatch cold), separato dal resto
# della telemetria (file dedicato, non tocca host_perf.csv/gpu_telemetry.csv
# gia' in esecuzione).
$out = "C:\Users\Admin\Documents\GitHub\vMemoryFabric\logs\z8_local\telemetry_fase6a\pcie_throughput.csv"
"timestamp_iso,rxpci_mb_s,txpci_mb_s" | Out-File -FilePath $out -Encoding utf8

$deadline = (Get-Date).AddSeconds(7200)
while ((Get-Date) -lt $deadline) {
    $ts = (Get-Date).ToString("o")
    try {
        $line = & nvidia-smi dmon -s t -c 1 | Select-Object -Last 1
        $parts = ($line -replace '^\s+','') -split '\s+'
        if ($parts.Length -ge 3) {
            "$ts,$($parts[1]),$($parts[2])" | Out-File -FilePath $out -Append -Encoding utf8
        }
    } catch {}
    Start-Sleep -Seconds 5
}
