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
| Ascend 910B | AutoDL | Ascend 910B | — | Richiede **MindSpore**, non PyTorch/vLLM nativo — attrito reale, non un semplice swap di endpoint |
| Ascend 910B/910C | (canale diretto partner) | — | — | Fuori da questa mappa per scelta esplicita dell'autore — vedi conversazione, non documentato qui |

**Caveat**: prezzi e disponibilità trovati via ricerca al momento della
stesura, non confermati sulle pagine live dei singoli provider — trattare
come direzionali, verificare prima di impegnare budget reale. Nessuno di
questi canali è stato testato end-to-end da questo progetto.

## 3. Il punto operativo più utile

**Tenstorrent via Koyeb** è l'elemento con il maggior rapporto segnale/costo
di questa mappa: consente di testare la tesi Galaxy/multi-scheda di issue #8
(§"Roadmap hardware multi-device") in pay-per-use, con una configurazione
già a 4 schede meshate (TT-Loudbox), prima di qualunque decisione
d'acquisto. Resta valido il caveat tecnico già registrato in issue #8: lo
stack attuale (`GPUTransfer`, `EAT`/`Tier`) è CUDA-specifico e andrebbe
astratto comunque, indipendentemente da dove si affitta l'hardware.

## 4. Manutenzione

Questo documento è dichiaratamente instabile nel tempo (prezzi/provider
cambiano su base mensile) — a differenza di `silicon_landscape_watch.md`
(che punta a paradigmi architetturali, molto più stabili) e di issue #8
(che punta a decisioni di roadmap, stabili fino alla prossima revisione
esplicita). Da aggiornare quando si valuta seriamente un test reale su uno
di questi canali, non su base calendarizzata.
