# Related Work — L'offloader di pesi nativo di vLLM vs EMH: il vero vicino, e la distanza misurata in quattro assi

**Status:** Nota di ricerca / posizionamento, **verificata sul codice sorgente** di vLLM
(clone, HEAD `18c5372` del 2026-09-01; presenza per release verificata sui tag). Non è un
report sperimentale. Nasce dalla review di `related_work_elastic_ep.md`: quel report
posiziona OSX contro Elastic Expert Parallelism, ma **il concorrente interno a vLLM è un
altro** e lì non è nominato. Questo documento chiude il buco. È il deliverable S0-bis di
issue #68; il piano di misura che ne discende è
[`test_plan_emh_vs_vllm_offloader.md`](test_plan_emh_vs_vllm_offloader.md).

**Data:** 2026-09-02
**Progetto:** OSX — Operating System for Experts (repo: `vMemoryFabric`)
**Scope:** cosa fa davvero `vllm/model_executor/offloader/`, cosa non può fare per
costruzione, e cosa questo cambia per il posizionamento di EMH nel paper (Filone B).

**Vedi anche:** [`related_work_elastic_ep.md`](related_work_elastic_ep.md) (EEP, asse
orizzontale), [`related_work_petals_exllama.md`](related_work_petals_exllama.md),
`logbook_paper.md` (entry 2026-08-13: FineMoE resta il vicino *accademico*; questo è il
vicino *ingegneristico*, ed è dentro l'engine da cui dipendiamo).

---

## 1. Cosa c'è, da quando

`vllm/model_executor/offloader/` — quattro file: `base.py`, `uva.py`, `prefetch.py`,
`prefetch_ops.py` — governato da `vllm/config/offload.py` (`OffloadConfig`,
`offload_backend ∈ {auto, uva, prefetch}`).

Presenza verificata sui tag: **assente in v0.13.0** (2025-12-18), **presente in v0.25.0**
(2026-07-08) e in tutte le successive fino a v0.28.0 (2026-08-23). Il punto esatto di
ingresso è fra le due; per il nostro scopo basta la conseguenza: **è fuori dalla finestra
pinnabile** `0.9.0`–`0.10.1` (`vllm_torch27_compat_analysis.md`) e vive nel motore V1,
mentre `GCSGWorker` eredita dal worker V0. Vale lo stesso vincolo strutturale di EEP:
**non può girare nello stesso processo di GCSG.**

Accanto, e adiacente: `vllm/v1/kv_offload/tiering/` — tiering multi-livello, ma **per la
KV cache**, non per i pesi. Non è un concorrente di EMH oggi; lo diventerebbe se il paper
reclamasse la generalità "hot AI objects" del README invece del perimetro reale del PoC
(esperti). Da tenere presente nella scelta delle parole.

## 2. Backend `uva` — zero-copy da pinned CPU, selezione per nome

`UVAOffloader` (`uva.py:21-160`) sposta i parametri in pinned CPU e sostituisce
`p.data` con una **vista CUDA della memoria host** via UVA
(`get_accelerator_view_from_cpu_tensor`, riga 118): la GPU legge direttamente attraverso
PCIe a ogni kernel che tocca il parametro. Nessun trasferimento esplicito, nessun buffer
GPU, nessun prefetch — il costo è pagato **sincronamente, a ogni forward, per ogni byte
letto**.

Due proprietà contano per il confronto:

- **Selezione per nome di parametro** (`cpu_offload_params`, `uva.py:100-108`), con match
  per segmento: l'esempio nel docstring di config è letteralmente
  `"experts.w2_weight"` che matcha `mlp.experts.w2_weight`. È pensato per gli esperti.
- **Budget in byte, riempito in ordine di iterazione** (`uva.py:82-84`, `:113`): i primi
  parametri incontrati vengono offloadati finché il budget non è esaurito. **Nessuna
  nozione di hotness**: quale esperto finisce su CPU dipende dall'ordine dei moduli, non
  dal traffico.

È l'erede diretto del `cpu_offload_gb` che GCSG già usa (#17) — con in più la selezione
per nome. Nient'altro.

## 3. Backend `prefetch` — gruppi di layer, schedule statico, buffer statici

`PrefetchOffloader` (`prefetch.py:127-372`) è la parte nuova e più seria.

**Selezione dei layer — statica e aritmetica** (`prefetch.py:184`):

```python
if module_index % self.group_size >= self.group_size - self.num_in_group:
```

Con `group_size=8, num_in_group=2` vengono offloadati i layer 6,7,14,15,22,23,… Deciso al
load, mai più rivisto. Non c'è nessuna dipendenza dal routing, dal carico, dalla
hotness: grep di `topk`, `router`, `gate`, `hot`, `lru`, `evict` su `prefetch.py`
restituisce **zero occorrenze**.

**Meccanismo a runtime** (`_hook_module_forward`, `prefetch.py:215-247`): il forward del
layer *i* viene avvolto da due custom op registrate nel grafo compilato —
`torch.ops.vllm.wait_prefetch(input, i)` prima, `torch.ops.vllm.start_prefetch(output,
i+step)` dopo. Il prefetch del layer `i+prefetch_step` parte quando finisce il layer *i*,
su un `copy_stream` dedicato, verso un **`StaticBufferPool`** (`prefetch.py:60-125`) di
buffer GPU pre-allocati e riusati circolarmente. Le `mutates_args` sui tensori creano le
dipendenze che `torch.compile` e CUDA graph rispettano — è ingegneria di qualità, ed è il
motivo per cui il compilatore può catturare il tutto in un grafo.

**Cosa viene copiato** (`start_onload_to_static`, `prefetch.py:509-551`): **tutti** i
parametri whitelisted del modulo, per intero, con `gpu_buffer.copy_(cpu_storage,
non_blocking=True)`. Nessun sottoinsieme, nessuna condizione.

## 4. Il fatto decisivo: la granularità è il layer, non l'esperto

Nel `FusedMoE` di vLLM i pesi di tutti gli esperti di un layer sono **un solo tensore**:

```python
# unquantized_fused_moe_method.py:70-79
w13_weight = torch.nn.Parameter(torch.empty(num_experts, ...))
layer.register_parameter("w13_weight", w13_weight)
```

L'offloader lavora **per parametro**. Quindi quando `offload_params={"w13_weight",
"w2_weight"}` seleziona gli esperti di un layer, seleziona **tutti gli 8** (Mixtral) — e
`start_onload_to_static` li copia tutti, anche se il router userà solo i top-2. Con `uva`
la lettura è on-demand e tocca solo i righe del tensore effettivamente indicizzate dal
kernel, ma senza prefetch e senza nessuna decisione di residenza.

Non è un'omissione che si corregge con una flag: **è il layout dei pesi.** Per muovere un
esperto singolo servirebbe un tensore per esperto, cioè un `FusedMoE` diverso — che è
precisamente ciò che l'EAT `(expert_id, shard_idx)` presuppone.

## 5. Confronto

| | `uva` | `prefetch` | EMH / OSX |
|---|---|---|---|
| Tier | 2 (pinned CPU ↔ VRAM) | 2 (pinned CPU ↔ VRAM) | 4 (VRAM / DDR4 / PMEM / NVMe) |
| Chi decide cosa risiede | ordine di iterazione fino a budget | `module_index % group_size`, al load | hotness (EAT) + gating predittivo (PT-PEP) + SEE |
| Routing-aware | no | **no** (zero occorrenze) | sì |
| Granularità | parametro (= tutti gli esperti del layer) | parametro (= tutti gli esperti del layer) | shard di esperto, 256 MB |
| Trasferimento | sincrono, on-demand, a ogni accesso | asincrono, `prefetch_step` layer avanti, nascosto se il compute copre | promote/evict asincroni, prefetch predittivo |
| Eviction dinamica | no | no (buffer circolari, schedule fisso) | sì |
| Byte mossi per token | ∝ esperti *instradati* dei layer offloadati | ∝ **tutti** gli esperti dei layer offloadati, sempre | ∝ esperti instradati **e mancanti** |
| Persistenza fra riavvii | no | no | #27 (in corso) |
| Integrazione compilatore / CUDA graph | sì | **sì, nativa** | no (`--enforce-eager`) |
| Qualità del modello | invariata (stessi pesi) | invariata | ≤ 2% MMLU target (shadow INT4) |

Le ultime due righe sono il prezzo che EMH paga e vanno dette nel paper con la stessa
chiarezza delle prime.

## 6. Cosa cambia per il posizionamento

1. **"Nessuno fa offload di esperti in vLLM" non è più vero**, e un reviewer che conosce
   vLLM lo dirà. Il claim difendibile è più stretto e più forte: *nessuno lo fa in modo
   routing-aware, a granularità di esperto, su più di due tier, con eviction dinamica.*
   Quattro assi, ciascuno verificabile con una riga di codice citata sopra.
2. **Il confronto sperimentale giusto è contro `prefetch`**, non contro `cpu_offload_gb`
   nudo. `prefetch` è ciò che un utente vLLM competente userebbe oggi per far entrare
   Mixtral in 24 GB; batterlo o pareggiarlo a parità di VRAM è la misura che conta.
   Batterlo dove il modello matematico dice che si deve (routing sparso, k ≪ E) e
   *perderci dove dice che si deve* (routing uniforme) è la validazione. Vedi il test
   plan.
3. **FineMoE resta il vicino accademico** (`logbook_paper.md`, 2026-08-13). Questo è il
   vicino ingegneristico. Sono due paragrafi diversi del Related Work e nessuno dei due
   sostituisce l'altro.
4. **Il perimetro delle parole**: `kv_offload/tiering` esiste. Se il paper dice "hot AI
   objects" in generale, qualcuno chiederà perché non confrontiamo anche quello. Il PoC è
   sugli esperti; il paper dovrebbe dirlo.

## 7. No-list

| Componente | Perché no | Rivalutare se |
|---|---|---|
| `StaticBufferPool` + custom op `wait/start_prefetch` | è la risposta giusta al problema "prefetch compatibile con CUDA graph"; OSX gira `--enforce-eager` | il PoC abbandona eager — a quel punto è il modello da copiare, non da reinventare |
| Schedule `module_index % group_size` | è esattamente ciò che EMH sostituisce | mai |
| UVA zero-copy come tier | è il "tier 1c senza promozione": leggere da DDR4 senza mai portare in VRAM | come **baseline** nel test plan, non come componente |

## 8. Fonti (HEAD `18c5372`, 2026-09-01; tag verificati v0.13.0, v0.25.0–v0.28.0)

- `vllm/config/offload.py` — `OffloadConfig`, `UVAOffloadConfig`, `PrefetchOffloadConfig`
- `vllm/model_executor/offloader/uva.py` — righe 21-160
- `vllm/model_executor/offloader/prefetch.py` — righe 60-125 (buffer pool), 127-372
  (offloader), 374-551 (`_ModuleOffloader`, `start_onload_to_static`)
- `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py:70-79` — layout
  `w13_weight (num_experts, …)`
- `vllm/v1/kv_offload/tiering/base.py` — tiering KV cache (adiacente)
