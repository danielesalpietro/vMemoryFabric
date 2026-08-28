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
