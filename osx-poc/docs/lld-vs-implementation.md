# LLD v1.0 vs. implementazione attuale — confronto

Confronto tra [OSX LLD v1.0](lld-osx-v1.0.md) / [dOSX LLD v1.0](lld-dosx-v1.0.md) e lo stato
reale del codice in `osx-poc/src`, `configs/osx_default.yaml`, `CHANGELOG.MD` al 2026-08-10
(release **Möllstorp**, v0.2.0-dev — Sprint 3 in corso).

## Discrepanze concrete (non solo "non ancora implementato")

### 1. EAT entry: 28 byte dichiarati, non ancora veri a runtime
`EATEntry` (`src/eat/types.py:32`) è una `@dataclass` Python — oggetti con overhead
dell'interprete, riferimenti, ecc. — non un layout binario compatto. Il commento nel codice
stesso lo dice esplicitamente: *"In Python usiamo dataclass per chiarezza; il layout compatto
verrà implementato con ctypes/struct nella fase di ottimizzazione"* (mai iniziata). Quindi:
**28 byte/entry è un target di design, non una proprietà misurabile oggi**. Se la LLD v1.0
la presenta come dato di fatto, va corretto il framing (target vs. stato).

### 2. "EMH 5 tier" — il repo ne documenta 4, di cui 3 attivi
README e config elencano: EMH-1a (VRAM), EMH-1c (DDR4), EMH-2 (PMEM, deferred), EMH-3
(NVMe). **Quattro**, non cinque. E solo tre sono attivi in dev (`tiers_active: [0, 1, 2]` in
`configs/osx_default.yaml` — PMEM escluso). Serve chiarire: la LLD v1.0 introduce un quinto
tier non ancora documentato nel repo (es. un layer di rete/fabric conteggiato come tier?),
oppure è un conteggio inconsistente da correggere nella LLD stessa. Non risolvibile dal codice
solo — richiede input.

### 3. AER "Base+LoRA-Delta" — il design è coerente, l'implementazione è zero
Punto positivo: qui LLD e codice **si allineano concettualmente**. Il docstring di
`aer.py` dice esplicitamente *"Base weights immutabili. LoRA Delta sync via PCIe"* — stesso
principio della LLD. Ma `replication_factor()` ritorna sempre `1` e `sync_lora_delta()` è
`pass` — nessuna replica è mai avvenuta, nemmeno in test. Utile sapere che il design è
validato concettualmente da due fonti indipendenti (LLD e stub), ma zero evidenza empirica.

### 4. Componenti della LLD OSX assenti dal repo
**ICP** (single-layer Commodity vs two-layer NVL72) e **IoX Fabric**: zero menzioni in
`src/`, README, config, changelog. Non sono "da implementare" nel senso di Sprint pending —
sono concettualmente nuovi rispetto a tutto ciò che esiste. Prima di scrivere codice per
questi due, servono le specifiche (§4 e §5 di [lld-osx-v1.0.md](lld-osx-v1.0.md), entrambe
`☐ TODO`).

### 5. dOSX è interamente non implementato — e presuppone hardware che il progetto non ha
Non un singolo modulo, test o riferimento a Coordinator/election/discovery/epoch/rack esiste
nel repo (verificato via grep — vedi [lld-dosx-v1.0.md](lld-dosx-v1.0.md)). Più
significativo: dOSX presuppone un ambiente **multi-rack** mentre il dev environment
dichiarato in README è esplicitamente **single-box, single-GPU** ("Dev environment
constraints" — dual-GPU è già "not yet available", figuriamoci multi-rack). Questo è lo
stesso pattern di scope-jump discusso in precedenza nella conversazione (OSX PoC → "sistema
operativo universale"): la distanza tra dOSX-LLD e ciò che è verificabile oggi è maggiore
di quella tra OSX-LLD e l'implementazione attuale.

## Cosa invece è già solido (M1/EAT)

A differenza di M2/M3/ICP/IoX/dOSX (tutti skeleton o pure design), **M1 (EAT) ha numeri
reali**, misurati due volte su hardware target reale via CI (`workflow_dispatch`), non
stimati: lookup p50 ≈ 2.6 µs / p99 ≈ 4.6 µs, ~177k insert/sec, con due problemi aperti
*quantificati* (Bloom filter più lento della baseline plain-dict di 5-14.6×; tail latency
sotto contesa +1.360×). Qualunque LLD che descriva l'EAT dovrebbe ancorarsi a questi numeri
invece che a stime, visto che esistono già.

## Riepilogo stato per componente

| Componente | LLD v1.0 | Codice | Gap |
|---|---|---|---|
| M1 — EAT | struttura 28B | dataclass, non packed | design vs. runtime |
| M1 — EAT (perf) | non specificata nella LLD | misurata, reale | LLD dovrebbe citare i numeri reali |
| M2 — Tier Manager | EMH 5 tier, latenze/flussi | 4 tier doc./3 attivi, tutti i metodi `NotImplementedError` | conteggio tier da chiarire + Sprint 2 non chiuso |
| M3 — PT-PEP/GCSG | soglie e pipeline coerenti | tutto `NotImplementedError`, Sprint 3 in corso | nessun gap di design, solo di implementazione |
| M3 — AER | Base+LoRA-Delta | stub coerente col design, mai eseguito | nessun gap di design, zero evidenza empirica |
| ICP | single/two-layer | assente | da specificare da zero |
| IoX Fabric | menzionata | assente | da specificare da zero |
| dOSX (intero layer) | 7 sezioni | assente, hardware multi-rack non disponibile | da specificare da zero, fuori scope hardware attuale |

## Prossimo passo suggerito
Le sezioni `☐ TODO` nei due documenti LLD sono l'elenco esatto di cosa manca per rendere
questi documenti completi. Il punto 2 (conteggio tier EMH) è l'unico che sembra un errore
piuttosto che un contenuto mancante — vale la pena chiarirlo per primo prima di scrivere
altro sopra.
