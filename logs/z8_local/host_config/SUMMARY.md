# Z8 locale — snapshot config host/Docker/GPU (2026-08-17, post-upgrade)

Stesso schema di `logs/sprint4_tekniska/pod_config/` e
`logs/malmo_runpod/pod_config/`, applicato qui alla macchina locale
dopo aver alzato l'allocazione WSL2/Docker (issue #33 LOGBOOK, sessioni
"continued 12"/"continued 13").

## Host reale

```
CPU:      2x Intel Xeon Gold 6244 @ 3.60GHz (8 core/16 thread ciascuno,
          dual-socket) — 16 core fisici / 32 thread logici totali.
          AVX-512F/BW/VL + AVX512_VNNI confermati.
RAM:      503.7 GB totali (wmic, verificato).
GPU:      NVIDIA GeForce RTX 3090, 24576 MiB VRAM, driver 610.74, CC 8.6.
```

## Allocato a WSL2/Docker (`.wslconfig`)

```
memory=444GB      (era 196GB)
processors=28     (era 14)
swap=8GB
```

Margine lasciato a Windows: ~60GB RAM, 4 thread/2 core CPU.

## Verificato dopo il riavvio (`wsl --shutdown` + restart automatico)

- `docker info`: 28 CPU, 436.8GiB RAM.
- `free -h` (dentro il container): 436Gi totali, 432Gi liberi al momento
  dello snapshot (nessun carico attivo).
- GPU: 827 MiB/24576 MiB VRAM usata al momento dello snapshot (~3.4%,
  residuo di sessioni precedenti — nessun processo pesante attivo).
- Suite pytest completa: 160 passed / 3 skipped, invariata dopo entrambi
  i riavvii.

## Uso host misurato PRIMA della decisione di alzare l'allocazione

RAM libera 464GB/503.7GB, CPU al 1% di carico (`Win32_PerfFormattedData_
PerfOS_Processor`) — nessuna pressione, nessun processo in competizione.
Motivo per cui si è deciso di allocare di più senza spegnere nulla,
invece di ottimizzare prima.
