# Issue #33 — confronto hardware: Z8 locale → RunPod RTX 3090/EPYC → RunPod H200/Xeon Platinum

**Data**: 2026-08-17
**Strumento**: `osx-poc/benchmarks/perf_test_hardware.py` +
`osx-poc/benchmarks/perf_tuning_report.py` (issue #33, "continued
19"/"continued 20"/"continued 21" in `LOGBOOK_ISSUE33.MD`)
**Dati grezzi**: `hw_z8.json`, `hw_pod.json`, `hw_h200.json` +
i rispettivi `tuning_report_*.json` in questa stessa cartella.
**Metodologia**: stesso identico matmul SwiGLU sintetico (2 matmul,
formula FLOPs condivisa) a dimensioni Mixtral reali
(`hidden=4096, intermediate=14336`), eseguito una volta per host su
CPU e GPU, nessuna ripetizione statistica multi-sessione — i numeri
vanno letti come un singolo campione per host, non una media
consolidata (vedi "Limiti" in fondo).

## 1. Baseline — Z8 locale (Xeon Gold 6244 + RTX 3090)

Lo Z8 è il punto di partenza: hardware dedicato (non virtualizzato/
condiviso), self-hosted runner del progetto, unico host di questo
confronto NON affittato a ore.

| | Valore |
|---|---|
| CPU | Xeon Gold 6244 (Cascade Lake) |
| Core cgroup | 28.0 |
| AVX | AVX2 ✓, AVX-512F ✓, AVX-512-VNNI ✓ (no AMX) |
| RAM cgroup | 431.2 GB |
| Banda RAM (copy) | **19.84 GB/s** |
| GPU | RTX 3090, 24GB VRAM |
| PCIe pinned (min misurato) | 11.26 GB/s |

## 2. RunPod RTX 3090 / AMD EPYC-class (no AVX-512)

Pod cloud, GPU identica allo Z8 (RTX 3090) ma CPU virtualizzata di
generazione precedente e priva di AVX-512 — il profilo già misurato
in "continued 17"/"continued 18"/"continued 20".

| | Valore |
|---|---|
| CPU | AMD EPYC-class (Zen3), no AVX-512 |
| Core cgroup | 27.2 |
| AVX | AVX2 ✓, AVX-512F ✗, AVX-512-VNNI ✗ |
| RAM cgroup | 116.0 GB |
| Banda RAM (copy) | **10.98 GB/s** |
| GPU | RTX 3090, 24GB VRAM |
| PCIe pinned (min misurato) | 20.44 GB/s |

## 3. RunPod H200 / Intel Xeon Platinum 8568Y+ (Emerald Rapids, AMX)

Target "ultima generazione" richiesto esplicitamente dal project
owner per confronto — GPU e CPU entrambe più recenti dei due host
sopra.

| | Valore |
|---|---|
| CPU | Xeon Platinum 8568Y+ (Emerald Rapids) |
| Core cgroup | **10.2** (12 vCPU nominali) |
| AVX | AVX2 ✓, AVX-512F ✓, AVX-512-VNNI ✓, **AMX** (int8/bf16/tile) ✓ |
| RAM cgroup | 233.5 GB |
| Banda RAM (copy) | **5.96 GB/s** |
| GPU | H200, 140.4GB VRAM |
| PCIe pinned (min misurato) | 29.97 GB/s |

## 4. Tabella comparativa — metriche misurate

Dimensioni reali (`hidden=4096, intermediate=14336`), GFLOPS a p50,
stessa operazione su CPU e GPU per ogni host (`_bench_matmul_gflops`,
condivisa tra le due sezioni — non due metodologie diverse).

| Metrica | Z8 (baseline) | RunPod RTX3090/EPYC | RunPod H200/Xeon Plat. |
|---|---:|---:|---:|
| CPU core cgroup | 28.0 | 27.2 | **10.2** |
| Banda RAM (GB/s) | **19.84** | 10.98 | 5.96 |
| CPU GFLOPS batch=1 | **22.97** | 11.66 | 3.32 |
| CPU GFLOPS batch=8 | 63.60 | 28.44 | 26.17 |
| CPU GFLOPS batch=32 | 229.89 | 34.75 | 57.25 |
| GPU GFLOPS batch=1 | 393.45 | 429.90 | **1888.89** |
| GPU GFLOPS batch=32 | 10621.44 | 11190.15 | 22480.45 |
| PCIe pinned min (GB/s) | 11.26 | 20.44 | **29.97** |

## 5. Tabella comparativa — verdetti derivati (`perf_tuning_report.py`)

| Metrica derivata | Z8 | RunPod RTX3090/EPYC | RunPod H200 |
|---|---:|---:|---:|
| Slowdown CPU/GPU previsto (batch=1) | **17.1x** | 36.9x | **569.7x** |
| Coerente col range noto (24-26x/44.6x)? | No — sotto | Sì | No — molto sopra |
| Batching signal (batch32/batch1, CPU) | 10.0x | 3.0x | 17.3x |
| Max expert CPU pool, path1/INT4 | 135 | 30 | 69 |
| Max expert CPU pool, path AWQ/fp32 | 18 | 4 | 9 |
| Thread consigliati (solo pool-build) | 28 | 27 | 10 |

## 6. Analisi

### 6.1 — Il gap CPU/GPU peggiora, non migliora, con l'hardware "di ultima generazione"

Passando dallo Z8/RTX3090 (17.1x) al RunPod RTX3090/EPYC (36.9x) fino
all'H200 (**569.7x**), il rapporto CPU/GPU non si comprime — si
allarga di quasi due ordini di grandezza. La causa non è un
regresso nella GPU (l'H200 è ~4.4-4.8x più veloce delle RTX 3090 allo
stesso batch=1, come atteso per il salto generazionale) ma il fatto
che il lato CPU di **questo specifico pod** non ha tenuto il passo —
anzi è il più lento dei tre host misurati in termini assoluti (3.32
GFLOPS contro 22.97 dello Z8).

**Implicazione diretta per issue #33**: l'ipotesi implicita "hardware
più recente = cpu-offload più conveniente" è falsificata dai dati
raccolti oggi. Su un pod H200/Xeon Platinum come questo, il cpu-offload
è ANCORA MENO conveniente in termini relativi rispetto ai pod già
misurati — non di più.

### 6.2 — La banda RAM predice il throughput CPU molto meglio della presenza di AVX-512/VNNI

| Host | AVX-512F/VNNI | Banda RAM (GB/s) | CPU GFLOPS batch=1 |
|---|---|---:|---:|
| Z8 | ✓/✓ | 19.84 | 22.97 |
| RunPod EPYC | ✗/✗ | 10.98 | 11.66 |
| RunPod H200 | ✓/✓ (+AMX) | 5.96 | 3.32 |

Se AVX-512/VNNI fosse il fattore dominante, l'H200 (che li ha
entrambi, più AMX) dovrebbe performare come lo Z8 o meglio — invece è
il più lento dei tre, e il suo unico tratto distintivo negativo è la
banda RAM più bassa misurata (5.96 GB/s, quasi 1/3 di quella dello
Z8). Il forward per-token è un GEMV a bassa intensità aritmetica
(ogni peso letto da RAM produce un solo output) — **memory-bandwidth-
bound per costruzione**, non compute-bound: esattamente la spiegazione
già isolata in "continued 17" con l'A/B sul thread count, qui
confermata su un TERZO host con una correlazione diretta banda-RAM/
GFLOPS invece che con un solo test A/B.

**Non ancora verificato**: se la banda RAM bassa sull'H200 sia un
limite intrinseco di questa allocazione vCPU (12 vCPU su un host più
grande e condiviso, quindi meno banda memoria/cache L3 per-core) o un
artefatto della singola misura — un secondo campionamento sullo stesso
pod chiuderebbe il dubbio, non fatto in questa sessione. Tracciato in
[issue #45](https://github.com/danielesalpietro/vMemoryFabric/issues/45).

### 6.3 — AMX non è misurato da questo benchmark, e potrebbe cambiare il quadro

Il pod H200/Xeon Platinum espone `amx_bf16`, `amx_int8`, `amx_tile` —
capacità dedicate all'accelerazione di matmul INT8/BF16/TF32. Il
benchmark usa `torch.randn(...)` (fp32 di default): **AMX non viene
mai esercitato** in nessuno dei numeri sopra. Il vero potenziale CPU di
questo host, su una precisione che AMX accelera davvero, resta
sconosciuto — è un limite dichiarato dello strumento attuale, non
un'affermazione implicita che "questo hardware non ha nulla da
offrire". Andrebbe misurato con un benchmark dedicato a INT8/BF16 prima
di chiudere il discorso su questo pod specifico. Tracciato in
[issue #45](https://github.com/danielesalpietro/vMemoryFabric/issues/45),
insieme al dubbio banda-RAM di §6.2.

### 6.4 — Il guadagno potenziale da batching varia per host, non è una costante

Il segnale "batching GEMV→GEMM" (rapporto GFLOPS batch32/batch1)
oscilla tra 3.0x (RunPod EPYC) e 17.3x (RunPod H200), con lo Z8 a
10.0x nel mezzo — nessuna relazione ovvia con core count o AVX. Se la
direzione "batching" verrà mai scelta per issue #33, andrà validata
sull'hardware target reale di produzione, non assunta costante da un
solo campione.

### 6.5 — RAM/VRAM: nessuno dei tre host è mai stato un vincolo

Il budget RAM CPU-pool (30-135 expert path1, 4-18 expert path AWQ a
seconda dell'host) supera ampiamente lo `shadow_pool_size=2` di
produzione in tutti e tre i casi — non è mai stato, né è oggi, il
collo di bottiglia. Stesso discorso per PCIe: nessuno dei tre host
mostra il tetto WSL2/GPU-PV noto (tutti "ok", banda pinned minima
11-30 GB/s) — sono tutti pod Linux reali o lo Z8 su Docker Desktop
nativo, mai lo stesso ambiente WSL2 problematico di "continued 16".

## 7. Conclusioni

1. **La causa del rallentamento cpu-offload resta la stessa isolata in
   "continued 17"**: il forward per-token è un GEMV memory-bandwidth-
   bound. Il confronto a tre host lo conferma con un pattern di
   correlazione (banda RAM ↔ GFLOPS) invece che con un solo test A/B
   su un host solo.
2. **"Ultima generazione" non implica automaticamente un cpu-offload
   più conveniente** — dipende dal bilanciamento RAM/CPU-cores
   dell'allocazione specifica, non solo dalla generazione del
   silicio. Su questo pod H200 il gap è peggiorato, non migliorato.
3. **AMX resta una domanda aperta**, non chiusa: il framework attuale
   non lo misura. Se emergerà l'esigenza di sapere se AMX cambia il
   quadro, serve un benchmark dedicato a BF16/INT8, non
   un'estrapolazione dai numeri fp32 qui sopra — tracciato in
   [issue #45](https://github.com/danielesalpietro/vMemoryFabric/issues/45).
4. **Nessuna delle direzioni ancora aperte per issue #33** (batching
   GEMV→GEMM, issue #27 EPM, refresh periodico 60s) cambia verdetto
   alla luce di questi dati — restano scelte di roadmap indipendenti
   da questo confronto, non impattate da esso.

## 8. Limiti di questa analisi

- **Un solo campione per host, non una media**: ogni numero è una
  singola esecuzione di `perf_test_hardware.py` (30 ripetizioni
  interne per il p50, ma un solo run del programma) — non una serie
  storica. Variazioni del 5-10% tra run sullo stesso host sono
  plausibili (osservato: due misure indipendenti sull'H200,
  573.3x e 569.7x di slowdown previsto, per la stessa identica
  configurazione).
- **AMX non misurato** (vedi §6.3) — il benchmark usa solo fp32.
- **Causa della banda RAM bassa sull'H200 non confermata** (vedi
  §6.2) — allocazione vCPU condivisa vs. caratteristica intrinseca
  del pod, nessuna delle due verificata a fondo.
- **GB10/DGX Spark (prossimo target pianificato)**: architettura a
  memoria unificata CPU+GPU, concettualmente incompatibile con la
  metodologia CPU/RAM/GPU/PCIe separata di questo framework — andrà
  ripensata la metodologia prima di lanciare lo stesso script
  invariato su quell'hardware, non solo eseguirlo.
