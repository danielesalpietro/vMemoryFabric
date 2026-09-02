# Piano dei test — EMH vs offloader nativo di vLLM: confrontabilità matematica e sistemistica

**Status:** piano pre-registrato, nessuna misura ancora eseguita. Le ipotesi in §8 sono
scritte **prima** dei numeri, con soglie di pass/fail e con il significato di un fallimento;
vanno lette come impegno, non come previsione ottimista.
**Data:** 2026-09-02
**Progetto:** OSX — Operating System for Experts (repo: `vMemoryFabric`)
**Base fattuale:** [`related_work_vllm_offloader.md`](related_work_vllm_offloader.md)
(cosa fa l'offloader, verificato sul codice) e
[`related_work_elastic_ep.md`](related_work_elastic_ep.md) (perché EEP non è un braccio).
**Alimenta:** paper Filone B, §6 Evaluation e §7 Limitations (`logbook_paper.md`,
entry 2026-09-02).

---

## 0. Perché un piano e non "un benchmark"

Il confronto onesto ha un ostacolo strutturale: l'offloader di vLLM vive nel motore **V1**
in release ≥ 0.25, `GCSGWorker` nel motore **V0** in 0.9.0–0.10.1. Non possono girare nello
stesso processo, e nemmeno nella stessa versione di vLLM. Un numero di throughput di EMH
messo accanto a uno di `prefetch` confronterebbe **due engine**, non due strategie di
memoria.

Il piano risolve il problema in tre modi che si rinforzano, ed è questo che lo rende
*confrontabile*:

1. **Un modello di costo comune** (§1): ogni sistema è un punto nello stesso spazio
   (byte mossi per token, tempo di stallo per token, footprint VRAM), con una formula che
   lo predice. La misura valida o smentisce la formula, non solo il numero.
2. **Normalizzazione** (§4): ogni braccio è riportato come rapporto rispetto al proprio
   baseline senza offload, *sulla stessa versione*. Il gap fra engine si misura a parte e
   si dichiara.
3. **Un livello senza vLLM** (§5, L0): le tre primitive di trasferimento — copia pinned
   H2D, lettura zero-copy UVA, `promote_live_tensor()` — misurate su tensori identici in
   torch puro. Lì l'engine non c'è, e il modello di costo si calibra.

## 1. Modello di costo comune

### 1.1 Notazione

| Simbolo | Significato | Mixtral 8x7B |
|---|---|---|
| $L$ | layer MoE | 32 |
| $E$ | esperti per layer | 8 |
| $k$ | esperti instradati per token (top-$k$) | 2 |
| $S$ | byte di un esperto per layer, nel formato servito | ≈ 176 M param → **≈ 90 MB in INT4** (stima, da verificare sul checkpoint) |
| $V$ | budget VRAM riservato agli esperti | variabile sperimentale |
| $c = V/(S\,E\,L)$ | frazione di esperti che entra in VRAM | variabile sperimentale |
| $B$ | banda effettiva host→GPU del tier (misurata a L0) | PCIe 4.0 x16 sul Z8 |
| $T_c(l)$ | tempo di compute del layer $l$ | misurato |
| $R_{l,t}$ | insieme degli esperti instradati al layer $l$ per il token $t$, $|R_{l,t}| = k$ | — |
| $L_{\text{off}}$ | layer offloadati da `prefetch` | $L \cdot (1-c)$ a parità di VRAM |
| $h$ | hit rate del prefetch predittivo (PT-PEP) sugli esperti *non* residenti | misurato |

### 1.2 Byte mossi per token di decode

**`prefetch`** — copia tutti gli esperti dei layer offloadati, ad ogni forward, sempre:

$$\beta_P = S \cdot E \cdot L_{\text{off}} \qquad \text{(costante, indipendente dal routing)}$$

**`uva`** — legge on-demand solo gli esperti instradati dei layer offloadati:

$$\beta_U = S \cdot k \cdot L_{\text{off}}$$

**EMH** — muove solo gli esperti instradati **e non residenti** al momento dell'uso:

$$\beta_E = S \sum_{l} \sum_{e \in R_{l,t}} \mathbb{1}[e \notin \text{VRAM}_t] \;=\; S \cdot k \cdot L \cdot \mu$$

con $\mu$ miss rate medio. Sotto routing uniforme e residenza casuale,
$\mu \approx (1-c)(1-h)$. Sotto routing *sbilanciato* (pochi esperti caldi), la residenza
guidata da hotness abbassa $\mu$ ben sotto $(1-c)$: **è l'intero vantaggio di EMH, ed è
una funzione dello skew del routing**, non una costante.

### 1.3 Tempo di stallo per token

`prefetch` nasconde la copia se il compute dei `prefetch_step` layer precedenti la copre:

$$\sigma_P = \sum_{l \in \text{off}} \max\!\Big(0,\; \frac{S E}{B} - \sum_{j=l-\text{step}}^{l-1} T_c(j)\Big)$$

`uva` non nasconde nulla — la lettura è sincrona nel kernel:
$\sigma_U \approx \beta_U / B_{\text{uva}}$, con $B_{\text{uva}} < B$ (accesso sparso, non
copia contigua; da misurare a L0).

EMH nasconde ciò che PT-PEP predice in tempo; lo stallo è sui miss non predetti:
$\sigma_E \approx S\,k\,L\,\mu\,/\,B + \varepsilon$, con $\varepsilon$ overhead di
bookkeeping (EAT lookup, GCSG dispatch — già misurati: `bench_route_forward.py`).

### 1.4 Predizioni falsificabili (indipendenti dall'engine)

- **P1** — $\beta_P / \beta_U = E/k = 4$ per qualunque $L_{\text{off}}$. Pura geometria.
  Se la misura non dà ≈ 4, **lo strumento è rotto**, non il sistema.
- **P2** — $\beta_E \le \beta_U$ sempre (l'insieme dei miss è un sottoinsieme
  dell'instradato); uguaglianza solo con $h = 0$ e $c = 0$.
- **P3** — a routing uniforme, $\beta_E \to \beta_U \cdot (1-c)$: **nessun guadagno oltre la
  capacità**. Se EMH "vince" qui, sospettare la misura.
- **P4** — con routing Zipf di esponente $\alpha$, $\beta_E$ decresce monotonicamente in
  $\alpha$; $\beta_P$ resta costante. Il gap si apre con lo skew.
- **P5** — break-even in byte: EMH batte `prefetch` sse $k \cdot \mu < E \cdot (1-c)$, cioè
  $\mu < 4(1-c)$ per Mixtral — vero per costruzione a $c=0$, sempre più stretto al crescere
  di $c$. **Il vantaggio si assottiglia quando la VRAM basta quasi**: va detto.

### 1.5 Qualità

Gli offloader servono **gli stessi pesi**: $\Delta Q = 0$ per costruzione. EMH con shadow
INT4 paga $\Delta Q \le 2$ punti MMLU (target README). Il confronto è quindi **a due
obiettivi**: byte/latenza *e* qualità. Un punto Pareto, non un vincitore.

## 2. Metriche — una definizione, tutti i bracci

| ID | Metrica | Definizione | Strumento (engine-agnostico dove possibile) |
|---|---|---|---|
| M1 | throughput decode | token/s in regime, batch fisso | client OpenAI-compat, conteggio server-side |
| M2 | TPOT | tempo per token di output, p50/p95/p99 | timestamp client per token in streaming |
| M3 | TTFT | time-to-first-token, p50/p95 | idem |
| M4 | **byte H2D per token** | byte host→GPU / token generati | `nvidia-smi dmon -s t` (PCIe RX) campionato; cross-check con `TierManager.stats()` (EMH) e CUDA event sul `copy_stream` (prefetch) |
| M5 | VRAM di picco | `torch.cuda.max_memory_allocated` + `nvidia-smi` | entrambi |
| M6 | stallo per token | M2 − tempo compute puro (A0 stesso engine) | derivata |
| M7 | qualità | MMLU 5-shot, stesso sottoinsieme, greedy | `scripts/eval_mmlu_gcsg.py` (già esistente) |
| M8 | solo EMH | `contamination_rate`, `activation_rate`, hit rate PT-PEP, distribuzione per tier | `GCSGGuard.stats()`, `TierManager.stats()` |

M4 è la metrica che il modello di costo predice direttamente: è quella su cui si
valida §1, prima ancora di guardare M1/M2.

## 3. Bracci

| Braccio | Sistema | Engine | Girabile oggi |
|---|---|---|---|
| **A0-V0** | vLLM 0.10.1, nessun offload | V0 | sì (se Mixtral AWQ + KV entra in 24 GB; altrimenti `max-model-len` ridotto, dichiarato) |
| **A0-V1** | vLLM ≥ 0.25, nessun offload | V1 | sì — la 3090 (sm_86) regge `torch>=2.7`; il blocco #8 riguarda solo la 5060 Ti |
| **A1** | vLLM ≥ 0.25, `offload_backend=uva`, `cpu_offload_params={w13_weight,w2_weight}` | V1 | sì |
| **A2** | vLLM ≥ 0.25, `offload_backend=prefetch`, sweep `(group_size, num_in_group, prefetch_step)` | V1 | sì |
| **A3** | vLLM 0.10.1 + `GCSGWorker` + EMH (`enable_cpu_offload`, TierManager) | V0 | sì |
| A4 | exllamav3 CPU MoE offload (riferimento esterno, già analizzato in Marstrand) | — | opzionale |

**EEP non è un braccio.** Richiede più GPU, motore V1, Ray, e risolve capacità di
throughput, non di memoria. Compare a L2 solo come *controllo negativo documentato*
(perché non gira) e per la misura di D7.

Due immagini Docker, non una: `V0` (pin attuale) e `V1` (`torch>=2.7`/cu128 +
vLLM ≥ 0.25). La seconda **non tocca il pin di produzione**.

## 4. Normalizzazione — il cuore della confrontabilità sistemistica

Per ogni metrica $M$ e braccio $X$:

$$\rho_X = \frac{M(A_X)}{M(A0\text{ della stessa versione})}$$

Si confrontano i $\rho$, non i valori grezzi. Il gap $M(A0\text{-}V1)/M(A0\text{-}V0)$ si
riporta **a parte**, esplicitamente, in una riga sua. Intervalli di confidenza sui rapporti
via bootstrap (≥ 1000 ricampionamenti sulle run). Nessun rapporto viene pubblicato senza
il suo intervallo.

## 5. Livelli

### L0 — Primitive, torch puro, senza vLLM — *eseguibile ora*

Estende `bench_tier.py` e `bench_pcie_bandwidth_wsl2.py`. Tensori identici ($S$ byte,
stesso dtype, stesso layout), tre primitive:

| Primitiva | Rappresenta | Cosa misura |
|---|---|---|
| copia pinned → buffer GPU statico, `non_blocking` su stream dedicato | `prefetch` | $B$, latenza a vuoto, overlap con un kernel di compute concorrente |
| kernel che indicizza $k/E$ righe di un tensore UVA | `uva` | $B_{\text{uva}}$ effettiva sotto accesso sparso |
| `TierManager.promote_live_tensor()` DDR4 → VRAM | EMH | $B$ + $\varepsilon$ bookkeeping |

**Output:** le costanti del modello di costo ($B$, $B_{\text{uva}}$, $\varepsilon$) e la
verifica di **P1 a livello di primitiva**. Senza L0, §1 è un'ipotesi; con L0 è calibrato.

### L1 — Serving, singola GPU — *eseguibile ora, tutti i bracci*

Tre carichi, ciascuno con sweep del budget VRAM $c \in \{1,\ 0.75,\ 0.5,\ 0.25\}$:

- **W1 — MMLU 5-shot** (esistente): qualità M7 + latenza M2/M3. Routing "naturale",
  moderatamente sbilanciato.
- **W2 — decode sintetico** a lunghezza fissa: throughput M1 in regime, byte M4.
- **W3 — sweep di skew**: prompt raggruppati per entropia di routing. Non si può forzare
  il routing dentro vLLM; si può **selezionare**: una run di profilazione registra i top-$k$
  per token (i hook GCSG vedono già il gating), si calcola l'entropia per prompt, si
  costruiscono bucket (uniforme / medio / sbilanciato). È il carico che testa **P3 e P4** —
  cioè l'unico che può *smentire* EMH nel modo previsto dal modello.

### L2 — Dual-GPU eterogeneo — *bloccato da #8*

- D7: budget KV cache sotto regola "min globale" vs per-device.
- AER (se sopravvive a D5): replica asimmetrica 24 GB / 16 GB.
- EEP come controllo negativo: documentare l'errore con cui rifiuta di partire.

## 6. Controlli di equità

- Stesso checkpoint (Mixtral 8x7B AWQ INT4, quello del PoC), stesso `max-model-len`,
  stesso `max-num-seqs`, greedy, stessi prompt, stessi seed.
- Warm-up escluso (lezione di #3), ≥ 5 run per cella, mediana + IQR.
- Pinned memory in tutti i bracci (`uva`/`prefetch` la pinnano; EMH la usa su Linux —
  `GCSGWorker._should_pin_transfers()`).
- Affinità NUMA fissata (#49), fingerprint hardware prima di ogni sessione (#18,
  `perf_test_hardware.py`), archiviazione grezza con header comando/host/timestamp
  (convenzione `logs/`).
- **Eager vs grafo**: A2 usa CUDA graph nativamente, A3 gira `--enforce-eager`. A0 si
  esegue **in entrambe le modalità** per separare il costo di eager dal costo della
  strategia di memoria. Senza questo, un vantaggio di `prefetch` in M2 è inattribuibile.

## 7. Minacce alla validità — da scrivere nel paper, non da scoprire in review

1. **Confondimento di versione** (V0 vs V1). Mitigato da §4 e L0; non eliminato. Si dice.
2. **Granularità dello shard.** 256 MB contro ≈ 90 MB per esperto INT4: uno shard porta
   2–3 esperti, muoverne uno muove i vicini. Va parametrizzato (64 / 128 / 256 MB) o il
   confronto a granularità di esperto è solo nominale.
3. **Qualità asimmetrica.** Gli offloader non toccano i pesi, EMH sì (shadow INT4).
   Riportare sempre M7 accanto a M4: nessun grafico di byte senza il suo punto di qualità.
4. **Leakage PT-PEP.** Il classificatore è addestrato su dati MMLU-like; su W1 il hit
   rate è ottimista. W2/W3 sono il controllo.
5. **Strumento di M4.** I contatori PCIe di `nvidia-smi` campionano; il cross-check con
   i contatori interni è obbligatorio, e P1 è il test dello strumento.
6. **Capacità di A0.** Se Mixtral AWQ + KV non entra in 24 GB senza offload, A0 esiste
   solo a `max-model-len` ridotto: il rapporto $\rho$ è comunque definito, ma il regime è
   dichiarato.

## 8. Ipotesi pre-registrate

| ID | Ipotesi | Soglia | Se fallisce, significa |
|---|---|---|---|
| **H1** | L0: rapporto byte `prefetch`/`uva` per primitiva | $4 \pm 10\%$ | lo strumento M4 è inaffidabile: **fermarsi** |
| **H2** | L1-W1, $c = 0.5$: $\beta_E \le 0.5\,\beta_P$ | rapporto M4 ≤ 0.5 con IC al 95% sotto 0.6 | la residenza guidata da hotness non cattura lo skew reale di MMLU |
| **H3** | L1-W1, $c = 0.5$: TPOT p95 di EMH ≤ 1.2 × `prefetch` (tolleranza per eager, quantificata da A0 eager/grafo) | $\rho_{M2} \le 1.2$ | il vantaggio in byte non si traduce in latenza: $\varepsilon$ o miss non predetti dominano |
| **H4** | M7: $\Delta$MMLU EMH vs A0-V0 | ≤ 2 punti | il costo di qualità supera il target README; il Pareto si sposta |
| **H5** | L1-W3 bucket uniforme: guadagno EMH su `uva` in M4 | ≤ 10 % | **se EMH vince qui, la misura è sbagliata** (P3) |
| **H6** | L1-W3: $\beta_E$ monotona decrescente in skew, $\beta_P$ piatta | Spearman < −0.9 su ≥ 4 bucket | il modello §1 non descrive il sistema |
| H7 | L2 (dopo #8): KV budget "min globale" < per-device sul rank grande | misura diretta, nessuna soglia | — |

H1 e H5 sono **test dello strumento e del modello**, non del sistema: se falliscono non si
pubblica nulla del resto.

## 9. Ordine e costo

1. **L0** — 1–2 giorni, ora, sulla 3090. Sblocca §1.
2. **Immagine V1** — un giorno: `torch>=2.7`/cu128 + vLLM ≥ 0.25, separata dal pin.
3. **L1-W2** (throughput) → **L1-W1** (qualità + latenza) → **L1-W3** (skew). Circa una
   settimana di macchina, con le run notturne.
4. **L2** — quando #8 si sblocca. Non blocca il paper: L0 + L1 bastano per §6.

## 10. Cosa non misuriamo, e perché

- **Scaling orizzontale di EEP**: problema diverso, hardware assente. Detto in §7 del
  paper, non misurato.
- **PMEM/NVMe nei bracci vLLM**: non li hanno. Il confronto è a parità di **VRAM** con
  DDR4 come secondo tier comune; PMEM/NVMe sono esperimenti *aggiuntivi* solo-EMH, con
  A3 stesso come baseline (già in `bench_pmem_tier.py`).
- **KV cache tiering** (`vllm/v1/kv_offload/tiering`): fuori perimetro del PoC. Il paper
  deve dire "esperti", non "AI objects", o questa riga diventa una domanda del reviewer.
