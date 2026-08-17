# Issue #33 Fase 6a - host telemetry sampler (CPU/RAM/GPU), timestamped
# per riga per poter mappare i campioni alle fasi del run (16/32/64/full).
$out = "C:\Users\Admin\Documents\GitHub\vMemoryFabric\logs\z8_local\telemetry_fase6a\host_perf.csv"
"timestamp_iso,cpu_pct,ram_used_gb,ram_free_gb,ram_total_gb" | Out-File -FilePath $out -Encoding utf8

$gpuOut = "C:\Users\Admin\Documents\GitHub\vMemoryFabric\logs\z8_local\telemetry_fase6a\gpu_telemetry.csv"
"timestamp_iso,name,util_gpu_pct,util_mem_pct,mem_used_mib,mem_total_mib,power_w,temp_c" | Out-File -FilePath $gpuOut -Encoding utf8

$deadline = (Get-Date).AddSeconds(7200)
while ((Get-Date) -lt $deadline) {
    $ts = (Get-Date).ToString("o")

    $cpu = (Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'").PercentProcessorTime
    $os = Get-CimInstance Win32_OperatingSystem
    $ramTotalGb = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
    $ramFreeGb  = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
    $ramUsedGb  = [math]::Round($ramTotalGb - $ramFreeGb, 2)
    "$ts,$cpu,$ramUsedGb,$ramFreeGb,$ramTotalGb" | Out-File -FilePath $out -Append -Encoding utf8

    try {
        $gpuLine = & nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
        if ($gpuLine) {
            "$ts,$gpuLine" | Out-File -FilePath $gpuOut -Append -Encoding utf8
        }
    } catch {}

    Start-Sleep -Seconds 10
}
