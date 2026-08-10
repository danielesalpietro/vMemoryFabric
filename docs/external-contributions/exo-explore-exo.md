# exo-explore/exo

**Repo:** https://github.com/exo-explore/exo
**Valutato il:** 2026-08-10
**Licenza:** Apache-2.0 · **OSX è MIT** — compatibili, nessun blocco legale a
riprendere pattern o codice con attribuzione.

## Cos'è

"Run frontier AI locally" — connette più device (principalmente Apple
Silicon) in un cluster di inferenza, con discovery automatico, sharding
topology-aware di modelli di grandi dimensioni, tensor parallelism e RDMA su
Thunderbolt. Stack: Python (`src/exo/`) su backend MLX/Metal, più binding
Rust (`rust/exo_rs`, `rust/networking`) per networking a bassa latenza
(libp2p) e RDMA.

Guardato perché il problema che risolve — piazzare/spostare pezzi di modello
su risorse eterogenee tenendo conto di memoria e banda disponibili — è
concettualmente adiacente a quello che OSX risolve per gli esperti MoE
all'interno di un singolo nodo (EMH: VRAM/DDR/PMEM/NVMe).

## Rilevante da vicino — M2 Tier Manager (Sprint 2, Eketorp — prossimo)

- [`src/exo/download/shard_downloader.py`](https://github.com/exo-explore/exo/blob/main/src/exo/download/shard_downloader.py)
  + `impl_shard_downloader.py` + `coordinator.py`: downloader di shard
  asincrono, cancellabile, con progress callback e semantica "overlap → la
  vecchia download viene cancellata e riparte quella nuova". È `asyncio`
  puro, **non legato a MLX/Metal** — a differenza della maggior parte del
  resto del repo, è effettivamente portabile senza riscrittura pesante.

  È il problema più vicino a `src/tier/io.py` (`AsyncNVMeIO`), che oggi in
  OSX è ancora un proxy minimale su `aiofiles` (sostituto di `io_uring` per
  via dei vincoli Docker-on-Windows/WSL2, vedi `osx-poc/README.md` §"Dev
  environment constraints"). exo ha già affrontato, con test reali, i casi
  che M2 dovrà gestire:
  - `tests/test_cancel_download.py` — cancellazione a metà trasferimento
  - `tests/test_rate_limit_handling.py` — backpressure/rate limiting
  - `tests/test_re_download.py` — ripresa dopo fallimento
  - `tests/test_download_verification.py` — integrità dello shard scaricato

  **Azione consigliata:** leggere questi file come riferimento di design
  prima di implementare la logica di promotion/eviction async in
  `TierManager` (`src/tier/manager.py`), non necessariamente portare codice
  1:1.

## Rilevante a medio termine — M3 / AER (oggi stub single-GPU)

- [`src/exo/master/placement.py`](https://github.com/exo-explore/exo/blob/main/src/exo/master/placement.py)
  + `placement_utils.py`: placement topology-aware — calcola "cicli" di
  device connessi, filtra per memoria disponibile
  (`filter_cycles_by_memory`), pesa i candidati per quanto di un modello è
  già scaricato su ciascun nodo (`_cycle_download_score`) per evitare
  ridownload inutili in fase di scheduling.

  È l'analogo più vicino a ciò che **AER** (Adaptive Expert Replication,
  `src/scheduler/aer.py`) dovrà fare quando smette di essere uno stub — oggi
  bloccato su "dual-GPU setup" (vedi tabella vincoli in
  `osx-poc/README.md`).

- `rust/networking` (`discovery.rs`, `swarm.rs`, libp2p) + RDMA-over-
  Thunderbolt: risolve esattamente il gap elencato nei vincoli dev di OSX
  ("Dual GPU (RTX 5080) — non ancora disponibile", "AER replication — stub,
  single GPU"). **Non portabile direttamente** (Rust + libp2p +
  Thunderbolt-specific, mentre OSX target è Linux bare-metal/Windows +
  CUDA), ma il pattern di discovery automatico multi-nodo è quello che
  servirà quando arriverà la seconda GPU.

- [`src/exo/worker/engines/mlx/auto_parallel.py`](https://github.com/exo-explore/exo/blob/main/src/exo/worker/engines/mlx/auto_parallel.py)
  + `src/exo/worker/plan.py`: logica che decide come spezzare un modello su
  più device in base a risorse disponibili e banda/latenza di rete tra i
  link. Concettualmente è la versione multi-device di quello che **GCSG**
  (`src/scheduler/gcsg.py`) dovrà decidere per il piazzamento degli shard
  degli esperti — non portabile come codice (MLX-specific), utile come
  riferimento d'algoritmo (grafo di topologia → vincoli di capacità →
  scoring del placement).

## Non rilevante

- Il motore di inferenza (`worker/engines/mlx/*`, Rust bindings): non
  riusabile — OSX è CUDA/vLLM, exo è Metal/Apple Silicon. Stack di silicio
  incompatibili, nessun percorso di porting sensato.
- `bench/` (metodologia, `prefill_decode_bench.py`, `exo_bench.py`): utile
  solo come ispirazione di stile. La cultura "risultati onesti, misurati su
  hardware reale" combacia con quella già in uso in `osx-poc/LOGBOOK.md` e
  `benchmarks/bench_eat.py`, ma non c'è codice riusabile — problema diverso
  (multi-nodo distribuito vs. singolo nodo multi-tier).
- Dashboard, app macOS, packaging (dmg/pyinstaller): fuori scope, prodotto
  end-user non componente di sistema.

## In sintesi

Il contributo concreto più immediato è leggere `shard_downloader.py` come
riferimento di design prima di scrivere la logica async di `TierManager` in
Sprint 2 — è codice `asyncio` puro, adattabile con sforzo modesto. Il resto
(placement, discovery, auto-parallel) ha valore soprattutto architetturale
per quando M3/AER smetterà di essere uno stub single-GPU: da tenere presente
come riferimento, non da implementare ora.

Nessuna azione di adozione codice pianificata al momento — solo lettura di
riferimento in vista di Sprint 2.
