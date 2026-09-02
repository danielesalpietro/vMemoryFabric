# Related Work — Elastic Expert Parallelism (vLLM) vs OSX/EMH: due assi ortogonali sullo stesso oggetto

**Status:** Nota di ricerca / posizionamento, **verificata sul codice sorgente** di vLLM
(shallow clone, HEAD `18c5372` del 2026-09-01 — vedi §6 per i percorsi). Non è un report
sperimentale: nessun benchmark è stato eseguito, né qui né su vLLM. Deliverable **S0** di
issue #68; le decisioni aperte che ne derivano si discutono in
[discussion #70](https://github.com/danielesalpietro/vMemoryFabric/discussions/70).

**Data:** 2026-09-01
**Progetto:** OSX — Operating System for Experts (repo: `vMemoryFabric`)
**Scope:** capire cosa fa davvero Elastic Expert Parallelism (EEP) di vLLM, dove tocca il
territorio di OSX/EMH, e cosa di quel meccanismo è per noi riusabile, irrilevante o già
risolto altrove.

**Vedi anche:** [`related_work_petals_exllama.md`](related_work_petals_exllama.md) — stessa
disciplina applicata a Petals ed ExLlamaV2/V3 (sub-progetto Marstrand, issue #32).

> **Avvertenza sulla v1.** Issue #68 è stata scritta da fonti secondarie (RFC upstream,
> blog, risultati di ricerca) prima di questa verifica. Quattro affermazioni di quella
> issue non reggono alla lettura del codice; sono corrette in **§5**, e una di esse
> **indebolisce l'ipotesi principale** su cui l'issue proponeva di lavorare. Chi legge
> #68 senza questo report ha in mano una descrizione sbagliata.

---

## 1. Cos'è EEP, come funziona davvero

EEP ridimensiona a caldo il numero di engine-core data-parallel di un deployment MoE.
Poiché in vLLM il gruppo EP è `data_parallel_size × tensor_parallel_size`
(`elastic_execute.py:629-633`), cambiare DP a runtime ridimensiona l'EP e forza una
ridistribuzione degli esperti.

### 1.1 Superficie API

Un solo endpoint POST, `/scale_elastic_ep`
(`vllm/entrypoints/serve/elastic_ep/api_router.py:27-80`), con payload
`{"new_data_parallel_size": int, "drain_timeout": int}` — `drain_timeout` default **120 s**
nell'endpoint HTTP, **300 s** nella firma Python di `AsyncLLM.scale_elastic_ep`
(`vllm/v1/engine/async_llm.py:1119-1124`). Un secondo endpoint,
`/is_scaling_elastic_ep`, espone lo stato per il probing lato client
(`api_router.py:83-85`). Il drain delle richieste in volo **non è automatico**: avviene solo
se `VLLM_ELASTIC_EP_DRAIN_REQUESTS=1` (`vllm/envs.py:316`, usato in
`async_llm.py:1157-1158`).

### 1.2 Protocollo a due fasi: `prepare` poi `commit`

Il punto architetturalmente più interessante, e quello che i riassunti divulgativi
appiattiscono: lo scale **non è un'operazione atomica bloccante**, ma una macchina a stati a
due fasi (`vllm/distributed/elastic_ep/elastic_state.py:32-59`), guidata da
`async_llm._scale_elastic_ep()` che chiama in sequenza `prepare_elastic_ep()` e
`commit_elastic_ep()` (`async_llm.py:1137-1161`).

Per un rank già esistente in scale-up (`ScaleUpExistingEngineState`):

| Stato | Cosa fa | Blocca il forward? |
|---|---|---|
| `PREPARE` | costruisce gli standby group, stage dei quant method MoE, trasferisce i pesi densi ai nuovi rank | **no** |
| `SYNC_KV_CACHE_MEMORY_SIZE` | `all_reduce(MIN)` sulla memoria disponibile per la KV cache nel nuovo gruppo DP | **no** |
| `COMMIT_SCALE_UP` | distrugge il vecchio gruppo DP, commuta sul nuovo, EPLB reshuffle | **sì** (commentato nel sorgente: `# Blocks forward passes.`) |
| `COMPLETE` | — | — |

La fase `PREPARE` gira in un `ThreadPoolExecutor` dedicato
(`elastic_state.py:122-145`, `_execute_async`) **mentre il server continua a servire**; solo
il commit ferma i forward pass. È questo, non il drain, il meccanismo che rende lo scale
poco invasivo.

Due dettagli che contano per noi:

- **La KV cache viene rinegoziata al minimo globale.** `_sync_kv_cache_memory_size()`
  (`elastic_state.py:331-357`) fa un `all_reduce` con `ReduceOp.MIN` sulla memoria
  disponibile per la KV cache su tutto il nuovo gruppo DP. Il rank più povero detta il
  budget per tutti. Un nodo eterogeneo — esattamente il nostro caso d'uso — paga il minimo.
- **Le CUDA graph vengono rilasciate e ricatturate** (`_release_cuda_graphs`,
  `warm_and_capture`, `elastic_execute.py:402-417` e `711-746`), e il modello viene
  ricompilato con `model.compile(fullgraph=True, backend=...)` sul path di
  `switch_and_prepare()`.

### 1.3 Da dove arrivano i pesi di un nuovo rank — la scoperta più rilevante

Un nuovo engine-core **non legge mai i pesi da disco**. `ElasticEPScalingExecutor.load_model()`
è una riga: `self.worker.load_model(load_dummy_weights=True)`
(`elastic_execute.py:242-243`). Il modello viene costruito con pesi fittizi e poi riempito
via rete, ma **da due path distinti**:

1. **Pesi densi** (attention, embedding, norm): `batch_transfer_weights()`
   (`elastic_execute.py:62-101`) fa `isend`/`irecv` P2P batchati. La funzione **esclude
   esplicitamente i pesi degli esperti**, insieme a `expert_map` e `._shared_experts`
   (`elastic_execute.py:73-85`; il filtro è alla riga 84,
   `if param.data_ptr() not in expert_weights_set`). Il pairing sender→receiver è
   deterministico e bilanciato, con gestione del resto quando
   `num_new_workers % old_dp_size != 0` (`elastic_execute.py:296-314`, e il lato ricevente
   ricalcola lo stesso pairing in `prepare_new_worker()`, righe 647-664).
2. **Pesi degli esperti**: si muovono **solo** attraverso il reshuffle EPLB,
   `eplb_state.rearrange()` (`elastic_execute.py:568-602`, `eplb/eplb_state.py:725-729`). Il
   nuovo rank riceve prima la mappa `physical_to_logical` in broadcast
   (`receive_expert_mapping()`, righe 681-706), la espande alla nuova dimensione EP
   riempiendo di `-1` le posizioni non ancora assegnate, e solo dopo il reshuffle porta i
   tensori.

In scale-up il reshuffle è **asincrono**: `commit_scale_up()` chiama
`_perform_eplb_reshuffle(async_op=True)` (riga 612) — gli esperti si riorganizzano mentre il
servizio è già ripartito. Richiede però NIXL: con `--eplb-config.use_async=true` senza NIXL
installato la config solleva `ValueError` (`config/parallel.py:889-896`).

### 1.4 Scale-down

Simmetrico ma **sincrono**. `perform_scale_down_eplb_reshuffle()`
(`elastic_execute.py:625-637`) costruisce un `rank_mapping` che manda a `-1` ogni rank EP
oltre la nuova dimensione, e passa quella mappa a `rearrange()`. Il `-1` è il marcatore di
"rank che sparisce": `rebalance_execute.py:622-641` lo usa per rimappare gli indici degli
esperti prima della mossa. Gli esperti dei rank rimossi vengono quindi **ricollocati sui
rank superstiti**, non buttati; il rank che esce lo fa solo dopo
(`switch_and_remove()`, riga 418; il rank uscente notifica `SHUTDOWN_COMPLETE`).

### 1.5 Quanti esperti ci sono davvero, e cosa cresce quando si scala

Il conteggio va fatto **per layer MoE**, non per modello: Mixtral 8x7B ha 8 esperti per
layer × 32 layer = 256 istanze di esperto. EPLB opera per layer — tutti i suoi tensori hanno
shape `(num_moe_layers, num_physical_experts)` (`eplb/eplb_state.py:111`, `:168`).

La distinzione decisiva è fra esperti **logici** e **fisici**:

```
num_physical_experts     = num_experts + num_redundant_experts   # routed_experts.py:1074
num_local_physical_experts = num_physical_experts // ep_size     # rebalance_execute.py:494
assert num_physical_experts % num_gpus == 0                      # eplb/policy/default.py:131
```

Gli esperti *logici* sono quelli del modello (8 per layer in Mixtral) e non cambiano mai. Gli
esperti *fisici* sono gli slot su cui vengono materializzati, e un logico può occupare più
slot: sono le repliche (`num_redundant_experts`).

Qui sta il punto meno intuitivo di EEP, ed è esplicito nel codice
(`core_client.py:1641-1655`):

```python
num_redundant_experts = (
    num_physical_experts * new_data_parallel_size // cur_data_parallel_size
) - num_experts
```

**Lo scaling tiene costante il numero di slot fisici per rank e fa crescere il totale.** Il
surplus non è composto da esperti nuovi — non ne esistono altri — ma da **repliche di
quelli esistenti**, distribuite da EPLB in base al carico osservato. Su Mixtral 8x7B,
partendo da DP=2 senza ridondanza (8 slot fisici, 4 per rank):

| Scale | slot fisici/layer | di cui repliche | slot per rank |
|---|---|---|---|
| DP=2 (partenza) | 8 | 0 | 4 |
| DP=4 | 16 | 8 | 4 |
| DP=8 | 32 | 24 | 4 |

A DP=8 ogni esperto logico esiste in media in 4 copie. **Le GPU aggiunte non comprano
frammenti più piccoli: comprano repliche degli esperti caldi.** È coerente con il fatto che
EEP esista per assorbire picchi di traffico, non per far entrare modelli più grandi — per
quello serve TP o PP, non DP.

Ne segue anche il pavimento sullo scale-down: quando la formula darebbe
`num_redundant_experts < 0` i logici non ci starebbero più, e la chiamata fallisce con
`ValueError` indicando la dimensione minima, `ceil(num_experts × cur_dp / num_physical)`
(`core_client.py:1648-1655`). Da DP=4 con 16 slot fisici il minimo è 2.

Va detto per completezza che la replica più costosa in VRAM non è quella degli esperti: con
DP ogni rank tiene **una copia intera dei pesi densi** (attention, embedding, norm) — è
esattamente ciò che `batch_transfer_weights()` copia P2P a ogni nuovo rank (§1.3).

### 1.6 Vincoli reali a HEAD `18c5372`

Dalla validazione in `config/parallel.py:875-896` e da `config/vllm.py:2598-2599`:

| Vincolo | Dove | Natura |
|---|---|---|
| richiede `--enable-eplb` | `parallel.py:876-877` | **hard**, `ValueError` |
| `pipeline_parallel_size > 1` vietato | `parallel.py:878-882` | **hard**, `ValueError` |
| incompatibile con `data_parallel_external_lb` / `hybrid_lb` | `parallel.py:883-888` | **hard**, `NotImplementedError` — "relies on a single API server and core client" |
| EPLB async richiede NIXL | `parallel.py:889-896` | **hard**, `ValueError` |
| non supportato dal **V2 model runner** | `vllm.py:2598-2599` | **hard**, forza il runner V1 |
| `--data-parallel-backend ray` | `v1/engine/core_client.py:1636-1638` | **hard**, `assert` sul path di scale |
| `--tensor-parallel-size 1` | `tests/distributed/test_elastic_ep.py:167-168` | **de facto**, non validato nel codice |

La distinzione fra *hard* e *de facto* conta, ma va cercata in due posti diversi: la
config (`parallel.py`) **non** valida né Ray né TP, e questo trae in inganno. Il vincolo su
Ray è imposto altrove, sul path di scale: `DPLBAsyncMPClient.prepare_elastic_ep()` apre con
`assert parallel_config.data_parallel_backend == "ray"` — *"Only ray DP backend supports
scaling elastic EP"* (`core_client.py:1636-1638`) — e poco sotto pretende un
`CoreEngineActorManager`, che è l'actor manager Ray. **Ray è quindi un requisito hard**, solo
non dichiarato dove uno lo cercherebbe. Che i tre executor implementino tutti
`elastic_ep_execute` (`uniproc_executor.py:72`, `multiproc_executor.py:678`,
`ray_executor.py:378`) non basta a renderlo opzionale: lo scale non arriva mai fino a loro.

TP=1 invece resta *de facto*: nessuna assert lo impone su nessun path che ho letto, e
l'aritmetica EP tiene conto di `tp_size` (`elastic_execute.py:629-633`). Cosa funzioni
davvero con TP>1 è ignoto, non vietato.

Il test end-to-end (`tests/distributed/test_elastic_ep.py`) verifica accuratezza GSM8K
prima, dopo scale-up e dopo scale-down, con traffico attivo durante lo scale, e copre anche
il caso non divisibile 2→3.

---

## 2. Confronto con OSX / EMH

| Caratteristica | EEP (vLLM) | OSX / EMH (questo repo) |
|---|---|---|
| **Asse** | orizzontale: quanti rank, chi ospita quale esperto | verticale: in quale tier vive uno shard |
| **Unità** | esperto fisico per rank EP | shard `(expert_id, shard_idx)`, 256 MB (`eat/types.py:12-13`) |
| **Controller** | EPLB (`eplb_state.rearrange()`) | SEE policy, GCSG, AER (`tier/policies.py`, `scheduler/`) |
| **Trigger** | chiamata HTTP esplicita; autoscaling è milestone futura upstream | pressione VRAM, hotness, gating predittivo PT-PEP |
| **Movimento dati** | P2P NCCL (densi) + reshuffle EPLB (esperti) | promote/evict fra tier (`TierManager.promote()`, `manager.py:112`) |
| **Sorgente dei pesi** | **mai il disco** — dummy weights + rete (§1.3) | NVMe → PMEM → DDR4 → VRAM, disco incluso |
| **Stato persistente** | nessuno: la mappa esperti vive nel processo | EAT, con `version`/seqlock (`eat/types.py:51`) e hotness (`access_count`, `last_access_ts`) |
| **Granularità temporale** | evento raro (uno scale) | continua (ogni token, via gating) |
| **Topologia** | multi-rank, un solo API server | single-host oggi; multinodo fuori scope PoC |

I due sistemi non si sovrappongono per funzione — si sovrappongono per **risorsa** (la VRAM)
e per **oggetto** (l'esperto), descritto con due chiavi che non si parlano: EPLB ragiona in
`(logical_expert, physical_slot, ep_rank)`, OSX in `(expert_id, shard_idx, tier)`.

---

## 3. Cosa cambia per noi

### 3.1 L'EAT non ha nozione di rank — e a DP=1 non è un bug

`ExpertAccessTable` indicizza `(expert_id, shard_idx)` (`eat/eat.py:79`). Finché il
deployment ha un solo engine-core la chiave è completa per costruzione. Dopo un reshuffle
EPLB, invece, la stessa chiave descrive un esperto che quel rank può non servire più. Il
campo `version` esiste già come contatore seqlock (`eat/types.py:51`) ed è il posto naturale
dove agganciare un'epoch di mapping — ma è una decisione di modello dati, non un
refactoring: vedi **D1** e **D2** in discussion #70.

Nota di realismo: EEP non persiste **nulla** fra scale. La mappa esperti è ricostruita in
memoria a ogni riconfigurazione. La hotness accumulata dall'EAT è quindi informazione che
vLLM non ha e non conserva — è il nostro differenziale, ed è anche ciò che rischiamo di
buttare se scegliamo l'invalidazione totale in D2.

### 3.2 L'ipotesi "idratare dal tier warm" è più debole di come l'ha scritta #68

L'issue proponeva di servire i nuovi rank dal tier DDR4/PMEM invece che dal broadcast di
rete, assumendo implicitamente che l'alternativa fosse un caricamento da disco. **Non lo è**:
§1.3 mostra che il disco non viene mai toccato (`load_dummy_weights=True`), e che il
trasferimento è già P2P batchato su NCCL, sovrapposto al servizio nella fase `PREPARE`.

Il confronto reale è quindi **PCIe da DDR4/PMEM locale vs NCCL peer-to-peer intra-nodo**, e
su un nodo singolo con interconnessione decente il secondo vince quasi certamente. L'ipotesi
resta plausibile solo in due scenari, entrambi fuori dal nostro hardware attuale: (a)
multi-nodo con rete lenta fra host, (b) scale-up con più nuovi rank per singolo sender, dove
il sender diventa collo di bottiglia (§1.3, pairing 1→k). Questo **non chiude D4** — lo
riformula: la domanda non è più "il warm tier batte il disco" ma "esiste una topologia in
cui batte la rete", e la risposta va misurata, non assunta.

Un'opportunità sopravvive intatta ed è più interessante: **lo scale-down**. Gli esperti dei
rank rimossi vengono ricollocati sui superstiti (§1.4), il che comprime lo stesso numero di
esperti logici in meno VRAM. È esattamente la condizione di pressione che il TierManager
esiste per gestire — e vLLM, in quel momento, non ha nessun tier dove metterli.

### 3.3 AER ed EPLB: il confine è più netto di quanto temessi

EPLB opera su `physical_to_logical` con un numero fisso di slot fisici per rank e gestisce la
replica con i *redundant experts* (`--eplb-config.num_redundant_experts`, esposto
nell'esempio come `--re`). Attenzione però a §1.5: in EEP la replica **non è un extra
opzionale, è il meccanismo stesso dello scale-up** — le GPU aggiunte vengono riempite di
copie degli esperti caldi. EPLB è quindi un controller di replica a pieno titolo, non solo
un load balancer, e questo rende D5 meno accademica di come l'avevo posta.

AER (`scheduler/aer.py`) opera sul `replication_factor` fra device locali ed è oggi uno stub
deliberato che logga `WOULD_REPLICATE` senza replicare (single-GPU, issue #8). Non c'è
collisione **oggi** perché AER non alloca nulla. La collisione arriverebbe solo quando AER
diventa reale su dual-GPU — a quel punto due controller scriverebbero sullo stesso budget.
Vedi **D5**.

---

## 4. No-list — cosa non replichiamo, e quando rivalutare

Nella disciplina di #36: elencare esplicitamente ciò che si scarta, con la condizione che
riaprirebbe la decisione.

| Componente EEP | Perché no | Condizione di rivalutazione |
|---|---|---|
| Standby process group (`standby_state.py`) | risolve la ricostruzione di comunicatori NCCL/stateless; OSX non possiede comunicatori | se OSX arriva a coordinare più rank (fuori scope PoC, cfr. #35) |
| Pairing sender→receiver (`elastic_execute.py:296-314`) | aritmetica specifica del broadcast di pesi densi | mai, salvo si implementi un trasporto di pesi proprio |
| Rilascio/ricattura CUDA graph | il nostro path GCSG gira `--enforce-eager` nel dev setup | se il PoC adotta CUDA graph e cambia la residenza dei pesi a caldo |
| Comunicatore NIXL | dipendenza esterna, serve solo all'EPLB async | se si misura EEP reale e l'async EPLB diventa il path di riferimento |
| Reshuffle EPLB in sé | è policy di load balance fra rank, non di memoria | mai: è il complemento di OSX, non un'alternativa |
| `all_reduce(MIN)` sulla KV cache | conferma un pattern, non è codice da riusare | — (ma è un'osservazione da citare nel paper: penalizza l'hardware eterogeneo) |

Da monitorare upstream (aggancio a #37): milestone multi-nodo, autoscaling policy, e l'RFC di
disaccoppiamento da Ray
([vllm#28243](https://github.com/vllm-project/vllm/issues/28243)) — quest'ultimo perché una
gestione elastica esterna è l'unico punto in cui un componente come OSX potrebbe inserirsi
senza forkare vLLM.

---

## 5. Correzioni rispetto a issue #68 (v1, da fonti secondarie)

1. **"Backend Ray obbligatorio"** → **vero**, ma non dove lo si cerca. In `parallel.py` non
   c'è; l'`assert` sta sul path di scale, in `core_client.py:1636-1638`. Il punto sostanziale
   di #68 regge; è la mia prima stesura di questo report che sbagliava a derubricarlo a
   "path testato" dopo aver guardato solo la validazione di config (§1.6).
2. **"`tensor_parallel_size=1` obbligatorio"** → falso come vincolo. Il codice calcola
   `ep_size = dp_size × tp_size`. TP=1 è ciò che i test esercitano (§1.6).
3. **"La fase di reshuffle trasferisce i pesi degli esperti via NCCL"** → impreciso. Sono
   **due path distinti**: i pesi densi via `isend`/`irecv` P2P, che **escludono
   esplicitamente** gli esperti; gli esperti solo via reshuffle EPLB, per giunta asincrono in
   scale-up (§1.3).
4. **"Scale-up idratato dal tier warm invece che da broadcast"** → l'ipotesi era formulata
   contro un'alternativa sbagliata (caricamento da disco). Il disco non viene mai letto:
   `load_dummy_weights=True`. Il confronto vero è PCIe locale vs NCCL, e su nodo singolo
   probabilmente perdiamo (§3.2).

Non verificato, quindi da non affermare: #68 dichiarava "niente DBO". Nella validazione di
`parallel.py` **non** ho trovato un divieto esplicito EEP+DBO. Ho trovato che EEP e DBO
sono entrambi, separatamente, non supportati dal V2 model runner
(`config/vllm.py:2595-2599`). Se serve la risposta certa, va cercata nel path del runner V1,
non nella config.

---

## 6. Fonti (verificate su clone, HEAD `18c5372` del 2026-09-01)

Repository: `vllm-project/vllm`, shallow clone, commit
`18c53727cebb588771c5e2f62a207137b0d6dffd`.

File più rilevanti per verifiche future:

- `vllm/distributed/elastic_ep/elastic_state.py` — macchina a stati prepare/commit
- `vllm/distributed/elastic_ep/elastic_execute.py` — trasferimento pesi, reshuffle EPLB
- `vllm/distributed/elastic_ep/standby_state.py` — gruppi di comunicazione standby
- `vllm/entrypoints/serve/elastic_ep/api_router.py` — endpoint HTTP
- `vllm/v1/engine/async_llm.py` (righe 1119-1161) — orchestrazione lato engine
- `vllm/v1/engine/core_client.py` (righe 1629-1660) — vincolo Ray, contabilità degli
  esperti fisici/logici e pavimento dello scale-down
- `vllm/config/parallel.py` (righe 875-896) — validazione dei vincoli
- `vllm/config/vllm.py` (righe 2543-2599) — feature non supportate dal model runner V2
- `vllm/distributed/eplb/eplb_state.py`, `rebalance_execute.py`, `policy/default.py` —
  reshuffle, `rank_mapping`, contabilità fisici/logici
- `tests/distributed/test_elastic_ep.py` — path realmente esercitato in CI
- `examples/ray_serving/elastic_ep/` — script di lancio e di scaling

Fonti secondarie consultate (e in due punti smentite dal codice, §5): RFC upstream
[vllm#20323](https://github.com/vllm-project/vllm/issues/20323), RFC
[vllm#28243](https://github.com/vllm-project/vllm/issues/28243), blog vLLM
"Elastic Expert Parallelism in vLLM" (2026-05-14).
