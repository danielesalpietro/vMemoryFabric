# EfficientMoE/MoE-Infinity

**Repo:** https://github.com/EfficientMoE/MoE-Infinity
**Valutato il:** 2026-08-10
**Licenza:** Apache-2.0 · **OSX è MIT** — compatibili, nessun blocco legale a
riprendere pattern o codice con attribuzione.

## Cos'è

Libreria PyTorch per serving di modelli MoE con offload di esperti su
host/SSD, cache activation-aware e prefetch JIT (dispatcher C++ "Archer").
Guardata specificamente per capire se il problema bloccante di Sprint 3
(Oskarshamn) — [issue #10](https://github.com/danielesalpietro/vMemoryFabric/issues/10),
GCSG shadow pool su esperti `FusedMoE` AWQ-Marlin-packed, `CUDA error:
illegal memory access` nel kernel Marlin compilato — fosse già stato
affrontato altrove. Non lo era in modo identico, ma il repo ha tre pezzi
rilevanti quasi 1:1 con la forma del problema.

## Rilevante da vicino — Sprint 3 (Oskarshamn), issue #10 / GCSG shadow pool

- **Stesso bivio di design, stessa raccomandazione.**
  [PR #137](https://github.com/EfficientMoE/MoE-Infinity/pull/137) (RFC
  design-only, `docs/rfcs/gpt-oss-expert-offload.md`) affronta il problema
  gemello: eseguire un singolo expert quantizzato (MXFP4, non AWQ/Marlin,
  ma stesso vincolo — pesi "packed" che il path standard non sa gestire)
  fuori dal loro dispatcher normale. Propongono due opzioni identiche alle
  due direzioni annotate nel LOGBOOK di Sprint 3:
  - *Option A — dequant-on-copy a bf16 ("parity-first")*: dequantizzare
    l'expert al momento del fetch e riusare la GEMM bf16 esistente. Rischio
    basso, riusa un pattern già in produzione (GLM FP8:
    `SetScales` + `dequant_fp8_blockwise`).
  - *Option B — kernel nativo quantizzato ("memory-first")*: kernel Triton
    scritto in casa (`fused_mxfp4_gemm`), non il kernel del framework
    upstream.

  Raccomandazione esplicita del team: **A prima, B come ottimizzazione
  successiva** — "lowest risk, reuses proven machinery" contro "requires
  the dispatcher to carry blocks+scales together". Corrobora, da un team
  indipendente sullo stesso tipo di vincolo, la prima delle due direzioni
  già annotate per issue #10 (dequantizzazione AWQ/Marlin a mano). Il
  pattern GLM FP8 citato (`dequant_fp8_blockwise`) è un riferimento di
  design concreto da leggere prima di riscrivere la dequant AWQ/Marlin da
  zero.

- **Terza direzione non ancora annotata su issue #10: non chiamare il
  kernel Marlin di vLLM, vendorizzare Marlin GEMM a sé.**
  [PR #100](https://github.com/EfficientMoE/MoE-Infinity/pull/100)
  (`feat(kernel): fused attention/FFN/QKV kernels and Marlin GEMM`, merged)
  aggiunge `moe_infinity._marlin`, un'estensione CUDA con **Marlin INT4 GEMM
  vendorizzato** (non richiamato da vLLM), compilata AOT con **eager
  fallback** quando l'AOT non è disponibile. È strutturalmente il modo per
  evitare il problema che ha fatto crashare `_MarlinFusedShadowExpert`:
  se il crash origina dal fatto che `AWQMoEMethod`/`fused_marlin_moe` di
  vLLM assumono di essere chiamati dentro l'esecuzione batched normale di
  vLLM (workspace/`top_k` legati al contesto reale del modello, non a una
  chiamata isolata fuori-schedule), vendorizzare il kernel Marlin standalone
  e chiamarlo direttamente per un singolo expert bypassa quell'assunzione
  invece di provare a piegare l'API di vLLM a un uso che non supporta.
  **Non verificato nel dettaglio** (non ho letto il sorgente vendorizzato,
  solo la PR description) — da controllare prima di investire tempo, ma è
  un'alternativa concreta alle due già registrate ("dequant a mano" vs.
  "repro compute-sanitizer minimale").

- **Nessun precedente diretto sul crash stesso.** Cercato "Marlin" / "AWQ"
  / "illegal memory access" tra issue e PR del repo: zero risultati aperti
  pertinenti. Coerente con l'ipotesi sopra — non richiamando mai il kernel
  Marlin *di vLLM* per esecuzione isolata di un expert, MoE-Infinity non ha
  mai potuto incontrare questo bug specifico.

## Rilevante a medio termine — pattern di concorrenza per EAT/dispatcher

[PR #138](https://github.com/EfficientMoE/MoE-Infinity/pull/138)
(`fix(offload): enable offloaded MoE decode — ExpertDispatcher CAS race`)
corregge una race nel loro dispatcher C++: `ReplaceCacheCandidates` faceva
CAS `IDLE->FETCHING` in anticipo, in corsa con la transizione di stato del
dispatcher stesso, abortendo con `exec_state CAS failed`. Non applicabile
1:1 (OSX non ha un dispatcher C++ separato), ma conferma indirettamente
che la scelta già fatta per `EAT` — `RLock` per-entry invece di CAS
lock-free, verificata dai test di concorrenza in `tests/test_eat.py` — è
la strada più sicura per questa classe di problema: doppie transizioni di
stato concorrenti su un CAS lock-free sono facili da sbagliare anche in un
progetto maturo.

## Non rilevante

- Dispatcher C++ Archer nel suo complesso, prefetch JIT, activation-aware
  caching: architettura multi-modulo grande, non portabile a spezzoni senza
  riscrittura sostanziale. Da guardare eventualmente come riferimento
  d'insieme per M2/M3, non ora.
- DFlash (speculative decoding), route-ahead prefetch, supporto GLM-5.2 /
  Qwen3.5 / DeepSeek-V4: fuori scope, problema diverso da quello di OSX
  (serving multi-modello vs. tiering di memoria single-node).

## In sintesi

Nessun fix pronto da portare per issue #10, ma tre riscontri utili: (1) un
team indipendente, sullo stesso tipo di vincolo, raccomanda la stessa prima
direzione già annotata nel LOGBOOK (dequant a mano, "parity-first" prima di
un kernel nativo) — con un pattern di design concreto da leggere
(`dequant_fp8_blockwise`, GLM FP8 path); (2) un'alternativa non ancora
registrata — vendorizzare Marlin GEMM invece di dipendere dal kernel di
vLLM (`moe_infinity._marlin`, PR #100) — da verificare prima di investire
tempo; (3) nessun precedente del crash stesso, coerente con l'ipotesi che il
bug sia specifico all'uso del kernel Marlin *di vLLM* fuori dal suo
contesto di chiamata previsto.

Azione consigliata: aggiungere queste tre note come commento su issue #10
prima della prossima sessione su Sprint 3, così chi la riprende parte da
tre direzioni invece di due.
