# Cloud Procurement Map — dove eseguire la PoC su hardware non posseduto

**Data:** 2026-08-16
**Natura del documento:** mappa operativa di procurement, non architettura e non
roadmap decisionale. Distinta deliberatamente dagli altri due documenti dello
stesso filone, ciascuno con un ruolo diverso:

- [`silicon_landscape_watch.md`](silicon_landscape_watch.md) — *perché* un
  silicio è interessante (paradigma architetturale, collo di bottiglia
  attaccato). Letteratura per il paper, non decisioni di spesa.
- Issue [#8](https://github.com/danielesalpietro/vMemoryFabric/issues/8) —
  *quale* hardware entra in roadmap e in che ordine (RTX 5080 → AMD →
  Tenstorrent), incluso il rischio toolchain Blackwell/sm_120.
- **Questo documento** — *dove*, in pay-per-use, si può effettivamente
  affittare quell'hardware oggi per produrre dati reali per il paper, senza
  capitale immobilizzato in acquisto. Vive qui perché cambia più spesso di
  entrambi gli altri (prezzi e provider si muovono su base mensile) — va
  aggiornato a parte per non far scadere gli altri due documenti ogni volta
  che un provider cambia tariffa.

Origine: emerso da una domanda concreta ("esiste un RunPod-equivalente in
Cina con silicio cinese?") durante la discussione su issue #8 e sul silicon
landscape watch — utile abbastanza da meritare un posto proprio, non solo
una nota a margine.

**Revisione 2026-08-16 (round 2)**: da una critica esterna, verificata
punto per punto prima di accettarla — aggiunta §3 su egress/trasferimento
dati (RunPod $0, Koyeb $0.04/GB oltre 100GB/mese gratis, resto non
verificato), reso esplicito che il prezzo mancante di `TT-Loudbox` va
richiesto direttamente a Koyeb prima di spendere budget reale (§4),
corretta l'informazione su AutoDL/Ascend (non più MindSpore-only, ha
aperto una zona dedicata con `vllm-ascend`), spostata la riga sul canale
diretto partner fuori dalla tabella.

---

## 1. Perché esiste

Ricercatore indipendente, budget di conseguenza vincolato: il pay-per-use
consente di validare ipotesi hardware (in particolare Tenstorrent
Galaxy/multi-scheda per AER, issue #8) **prima** di un acquisto, e di
generare dati reali per il paper su hardware che altrimenti richiederebbe
capitale che non ha senso immobilizzare per un test.

## 2. Mappa (verificata 2026-08-16, non presa per buona da ricerche generiche)

| Silicio | Provider | Istanza | Prezzo (trovato) | Note |
|---|---|---|---|---|
| NVIDIA (già in uso) | RunPod | A100 SXM 80GB, RTX PRO 6000, ecc. | — | Provider già usato dal progetto — vedi `osx-poc/LOGBOOK.md`, pod Sprint 4 (episodio sm_120, vedi issue #8) |
| AMD MI300X | TensorWave | MI300X | **$1.71/hr** | Prezzo più basso trovato |
| AMD MI300X | RunPod | MI300X | $2.39/hr on-demand, **$1.49/hr spot** | Stesso account già in uso per NVIDIA |
| AMD MI300X | Hot Aisle | MI300X | ~$1.99–2.99/hr | |
| AMD MI300X | Oracle Cloud, DigitalOcean, Crusoe, Seeweb, Cirrascale | MI300X | variabile, non approfondito | |
| Tenstorrent Wormhole | **Koyeb** | TT-n300s (1x n300s, 24GB GDDR6, ~466 FP8 TFLOPS) | non trovato | Koyeb si dichiara l'**unico** cloud provider terzo per acceleratori Tenstorrent |
| Tenstorrent Wormhole | Koyeb | TT-Loudbox (4x n300s **meshati**, 96GB GDDR6, ~1864 FP8 TFLOPS) | non trovato | Configurazione già multi-scheda — testabile per l'ipotesi Galaxy/AER (issue #8 §"Roadmap hardware multi-device") senza acquisto |
| Ascend 910B | Luchentech Cloud (潞晨云) | NVIDIA + Ascend 910B stessa piattaforma | — | |
| Ascend 910B | 算力网 (Suanlix) | Ascend 910B dedicato | — | |
| Ascend 910B | 智启云川 (Zhiqi Cloud) | Ascend 910B | sconti fino al 50% su commitment pluriennali | |
| Ascend 910B | AutoDL | Ascend 910B (zona dedicata) | — | **Corretto 2026-08-16**: AutoDL ha aperto una zona dedicata Ascend 910B con deploy diretto via `vLLM` (`vllm-ascend`) — l'attrito non è più "riscrivere su MindSpore" (informazione precedente, superata). L'attrito reale è più sottile: `vllm-ascend` è un plugin con dipendenze CANN specifiche sotto il cofano, non un semplice swap di endpoint su vLLM vanilla |

**Caveat**: prezzi e disponibilità trovati via ricerca al momento della
stesura, non confermati sulle pagine live dei singoli provider — trattare
come direzionali, verificare prima di impegnare budget reale. Nessuno di
questi canali è stato testato end-to-end da questo progetto.

**Nota**: esiste anche un canale diretto verso Ascend 910B/910C tramite
partnership tecnologica dell'autore — deliberatamente non dettagliato qui
su richiesta esplicita, vedi conversazione. Non in tabella per non
introdurre una riga senza provider/prezzo/istanza in un documento
operativo pensato per essere scansionato rapidamente.

## 3. Egress e trasferimento dati — il costo nascosto

Spostare pesi di modelli (7B-13B+ o shard di esperti MoE) tra provider per
confronti comparativi può costare più del compute stesso se non
verificato prima. Dati confermati, non tutti i provider:

| Provider | Egress | Storage persistente | Note |
|---|---|---|---|
| RunPod | **$0** — nessun costo di egress verso internet | Network volume: $0.07/GB/mese (<1TB), $0.05/GB/mese (>1TB) | Vantaggio concreto: 5TB di pesi scaricati costano $0 qui contro ~$450 su AWS/~$600 su GCP. Caveat: alcuni host della Community Cloud possono avere costi di rete sottostanti non coperti da questa garanzia |
| Koyeb | 100GB/mese gratis, poi **$0.04/GB** | non verificato | Da mettere in conto se si spostano ripetutamente pesi verso/da un'istanza Tenstorrent |
| TensorWave, Hot Aisle, Luchentech, Suanlix, Zhiqi Cloud, AutoDL | **non verificato** | **non verificato** | Da controllare prima di un test reale — non presumere gratuità per analogia con RunPod |

**Implicazione pratica**: per confronti cross-provider (es. stesso shard di
esperti su RunPod/AMD vs Koyeb/Tenstorrent), il costo di trasferimento va
verificato quanto quello di calcolo — RunPod è gratis in uscita, Koyeb no,
gli altri sono ignoti.

## 4. Il punto operativo più utile

**Tenstorrent via Koyeb** è l'elemento con il maggior rapporto segnale/costo
di questa mappa: consente di testare la tesi Galaxy/multi-scheda di issue #8
(§"Roadmap hardware multi-device") in pay-per-use, con una configurazione
già a 4 schede meshate (TT-Loudbox), prima di qualunque decisione
d'acquisto. Resta valido il caveat tecnico già registrato in issue #8: lo
stack attuale (`GPUTransfer`, `EAT`/`Tier`) è CUDA-specifico e andrebbe
astratto comunque, indipendentemente da dove si affitta l'hardware.

**Prima di impegnare budget reale su questa istanza**: il prezzo di
`TT-Loudbox` non è pubblico (§2, "non trovato" non è un dato mancante da
completare via ricerca, è un limite reale della fonte) — va richiesto
direttamente a Koyeb (contatto/signup) prima di pianificare qualunque
test. Un conto è se costa $10/ora (compatibile con un budget da
ricercatore indipendente), un altro se costa $50/ora (richiederebbe
ripensare la strategia di test, es. partire dal singolo `TT-n300s` invece
del `TT-Loudbox` a 4 schede).

## 5. Manutenzione

Questo documento è dichiaratamente instabile nel tempo (prezzi/provider
cambiano su base mensile) — a differenza di `silicon_landscape_watch.md`
(che punta a paradigmi architetturali, molto più stabili) e di issue #8
(che punta a decisioni di roadmap, stabili fino alla prossima revisione
esplicita). Da aggiornare quando si valuta seriamente un test reale su uno
di questi canali, non su base calendarizzata.
