# dOSX — Low-Level Design v1.0 (layer distribuito)

> Status: **scaffold, zero implementazione**. Nessun file, modulo, test, config o riga di
> changelog nel repo menziona dOSX, coordinator, rack, epoch, o migration flow. Questo
> documento riporta solo la struttura delle sezioni comunicata; ogni contenuto tecnico
> (numero di step, dimensioni della tabella comparativa, dettagli epoch lifecycle) è
> `☐ TODO` perché non è mai stato specificato in questa conversazione né esiste altrove
> nel repo — va fornito, non dedotto.

---

## 0. Premessa di scope

dOSX presuppone un ambiente **multi-rack** (Coordinator con election/discovery, istanze
per-rack "OSX-NVL72"). L'attuale PoC OSX gira su un **singolo dev box** (Z8 G4, 1×RTX 3090,
Docker-on-Windows) — vedi `osx-poc/README.md`, tabella "Dev environment constraints". Non
esiste, al momento, alcun hardware multi-nodo nel progetto (AER stesso è stub per singola GPU,
in attesa della RTX 5080 per il caso *dual-GPU singolo nodo*, non multi-rack).

Questo non invalida il design — è legittimo progettare il layer distribuito prima di avere
l'hardware — ma va tenuto esplicito: **tutto ciò che segue è specifica, non è mai stato
eseguito né testato**, a differenza di OSX M1 (EAT) che ha numeri reali misurati su hardware.

---

## 1. dOSX Coordinator — election + discovery + global topology

☐ TODO. Da specificare:
- Algoritmo di election (Raft? Bully? altro?)
- Meccanismo di discovery (statico, gossip, service registry esterno?)
- Cosa contiene la "global topology" e chi la consulta (Expert Scheduler M3? Tier Manager M2 per decisioni cross-rack?)

---

## 2. OCS Inter-Rack Manager — epoch lifecycle

☐ TODO. Da specificare:
- Cosa delimita un epoch (tempo fisso? evento di rebalancing? cambio topologia?)
- Transizioni di stato dell'epoch lifecycle
- Come si relaziona con il `version` counter (CAS) già presente in `EATEntry` — è lo stesso concetto esteso al layer distribuito, o un meccanismo indipendente?

---

## 3. Per-Rack OSX-NVL72 Instance — EMH collassato + AER inter-rack + dOSX Agent

☐ TODO. In particolare da chiarire:
- "EMH collassato": i tier VRAM/DDR4/NVMe/PMEM esistenti (§EMH in OSX LLD) diventano un singolo tier logico per-rack visto dall'esterno? O è una gerarchia ridotta rispetto ai 4-5 tier locali?
- AER inter-rack: `AERManager` oggi (`src/scheduler/aer.py`) gestisce solo `replication_factor` e `sync_lora_delta` **dentro un singolo nodo** (stub, sempre `1`). Il salto a inter-rack è un'estensione dello stesso componente o un modulo nuovo?
- dOSX Agent: processo per-rack che parla col Coordinator (§1)? Non specificato.

---

## 4. Expert Migration Flow (8 step)

☐ TODO — i passi non sono stati forniti. Serve l'elenco degli 8 step per poter almeno
verificare la coerenza con il flusso di promozione/eviction già definito lato OSX
single-node (`TierManager.promote/evict`, oggi anch'esso non implementato — Sprint 2).

---

## 5. Metrics federati

☐ TODO. Base disponibile lato singolo nodo: Prometheus (porta 9090, scrape 5s),
`EAT.stats()`, stub `stats()` su Tier Manager/PT-PEP/GCSG/AER. Da definire: aggregazione
cross-rack (federation Prometheus? push gateway centralizzato?), e quali metriche del
Migration Flow (§4) vanno esposte.

---

## 6. Open Problems dOSX

☐ TODO — nessuno indicato. Un candidato ovvio da valutare, per analogia con quanto già
misurato lato OSX single-node: il Bloom filter dell'EAT non supporta cancellazione
(`remove_expert` → `NotImplementedError`); un layer di migrazione inter-rack che sposta
expert tra istanze aggraverebbe lo stesso problema (entry "fantasma" nel BF di più rack
contemporaneamente) se non viene risolto prima a livello locale.

---

## 7. OSX vs dOSX — tabella comparativa (8 dimensioni)

☐ TODO — le 8 dimensioni non sono state fornite. Scaffold minimo basato su ciò che è
certo dal repo (da espandere alle 8 dimensioni reali):

| Dimensione | OSX (single-node) | dOSX (multi-rack) |
|---|---|---|
| Hardware target dev | 1× RTX 3090, Z8 G4 | non disponibile nel progetto ad oggi |
| Stato implementazione | M1 rilasciato e benchmarkato; M2/M3 skeleton Sprint 2/3 | 0% — solo LLD |
| Scope Coordinator/election | n/a (single-node) | §1 |
| ... | | |

---

## Riferimenti verificati
Nessun file nel repo menziona dOSX. Assenza confermata via ricerca full-text
(`dOSX`, `mOSe`, `OCS`, `NVL72`, `Coordinator`) su `osx-poc/` — 2026-08-10.
