# Z8 G4 bare-metal — dati grezzi (2026-08-24)

Recuperati dalla macchina Z8 stessa il 2026-08-28 (branch `claude/z8-raw-data-recovery`),
mai committati al momento della sessione originale (`osx-poc/LOGBOOK_NEW_Z8.md`).
Verificati riga per riga contro i numeri già citati in prosa nel logbook prima del
commit (es. `gflops_at_p50 = 31.604908778112684` ↔ "31.60" citato nel logbook per
issue #33; `write_gbps`/`read_gbps`/`p50_us` di `bench_pmem_tier.json` ↔ i numeri PMEM
del logbook) — nessun mismatch.

## Nota tecnica: file `.json` vs `.clean.json`

I file `.json` originali (`hw_z8_baremetal.json`, `hw_z8_baremetal_numa0.json`,
`tuning_report_z8_baremetal.json`, `tuning_report_z8_baremetal_numa0.json`,
`bench_pmem_tier.json`) sono l'output **grezzo, byte-per-byte**, come catturato
sulla Z8 — non modificati, per lo stesso motivo per cui questo progetto non
sovrascrive mai dati sorgente. **Non sono JSON valido as-is**: il comando che li ha
generati ha catturato anche il banner di avvio del container CUDA
(`== CUDA ==`, versione, licenza NGC) prima del payload JSON vero e proprio
(che inizia alla prima riga `{`).

I file `*.clean.json` accanto a ciascuno sono la stessa identica struttura dati,
con solo il banner rimosso (`sed -n '/^{/,$p' file.json | python3 -m json.tool`),
pronti per essere caricati con `json.load()` da script/tooling (es. il cruscotto
dati del Filone B). Se serve rigenerarli, lo stesso comando produce lo stesso
risultato — nessuna trasformazione con perdita.

## File

| File | Issue | Contenuto |
|---|---|---|
| `perf_test_hardware_20260824/hw_z8_baremetal.json` | #33 | CPU/RAM/GPU/PCIe characterization, non-pinnato (32 core cross-NUMA) |
| `perf_test_hardware_20260824/hw_z8_baremetal_numa0.json` | #33/#49 | Stesso, NUMA-pinned nodo0 (16 core, mem locale GPU) |
| `perf_test_hardware_20260824/tuning_report_z8_baremetal.json` | #33 | Verdetto slowdown CPU-offload, non-pinnato |
| `perf_test_hardware_20260824/tuning_report_z8_baremetal_numa0.json` | #33/#49 | Stesso, NUMA-pinned |
| `pmem_tier_20260824/bench_pmem_tier.json` | #7/#45, EMH-2 | Raw bandwidth PMEM read/write, nvme→pmem P50, pmem→ddr4 P50 |
| `baseline_e32981d_20260824/hw_z8_baremetal_e32981d.json` | #33 | Run indipendente sul commit `e32981d` (confronto Z8/RunPod-EPYC/RunPod-H200), host container diverso — `gflops_at_p50` 30.16 vs. 31.60 del run principale: variazione run-to-run attesa, non un mismatch |
| `baseline_e32981d_20260824/tuning_report_z8_baremetal_e32981d.json` | #33 | Tuning report dello stesso run indipendente |
| `bench_20260824/test_output.log` | — | Log completo `docker compose run` pytest: 210 passed, 3 skipped, coverage 88% |
| `bench_20260824/bench_eat.log` | M1 | Benchmark EAT (10k entries/50k lookups), p50/p95/p99 in µs |
| `bench_20260824/bench_tier.log` | M2 | Benchmark TierManager (nvme_to_ddr4/ddr4_to_vram/promote), conferma `within_1.5x_theoretical_bandwidth_at_p50: true` |
| `telemetry_20260824_1253/gpu_telemetry.csv` | — | 696 campioni (10s), finestra 12:53–14:53 UTC — **quasi tutto idle**: solo 15/696 campioni con `util_gpu_pct` > 0, blip isolati da 10s, mai un plateau. Coerente col fatto che i benchmark issue #33 sono genuinamente CPU-bound: la GPU resta a riposo per la maggior parte della sessione |
| `telemetry_20260824_1253/host_perf.csv` | — | 696 campioni, `cpu_pct` max 28.2% — anche il carico host è modesto nel complesso della finestra |
| `telemetry_20260824_1253/pcie_throughput.csv` | — | 1415 campioni (5s) — throughput non-zero solo in 35/1415 (rx) e 92/1415 (tx) campioni, picchi 9065/9982 MB/s |
| `telemetry_20260824_1253/docker_stats.csv` | — | 36 campioni per-container — il segnale più denso della cartella: un container (`...-9035fcdab076`) mostra la RAM salire da 3.3 a 20.2 GiB in ~44s, poi ridiscendere |

Le 4 serie di `telemetry_20260824_1253/` non sono state usate per un grafico nel
cruscotto dati Filone B: campionate sull'intera finestra di sessione (2 ore), mentre
i benchmark reali occupano solo pochi minuti sparsi — un grafico ingenuo sarebbe una
linea piatta con 2-3 spike stretti, non informativo. Restano disponibili per chi
volesse ritagliare le finestre attive (coerenti con gli mtime dei JSON: 13:15-13:20,
13:41, 14:14) o usare `docker_stats.csv` per un grafico a densità di segnale più alta.
