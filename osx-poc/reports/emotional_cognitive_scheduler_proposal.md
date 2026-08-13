# Proposta di design: Emotional/Cognitive Tier Scheduler (ECS)

> **Stato: proposta di design, non implementata.** Nessun codice in questo
> documento è stato eseguito o verificato su hardware reale. Zero modifiche a
> `osx-poc/src/`. Segue lo stesso standard di onestà del resto del progetto
> (vedi README §"Known limitations"): qui si dichiara esplicitamente cosa è
> già reale nel codice, cosa è nuovo, e cosa resta una domanda aperta —
> niente è presentato come "quasi fatto" quando non lo è.
>
> Origine: discussione 2026-08-13 su un parallelismo biologico
> (adrenalina/attenzione/arte imparata/doti innate) applicato ai tier EMH.
> Questo documento prova a tradurre quel parallelismo in regole verificabili,
> ancorate al codice che esiste oggi in `osx-poc/src/`, invece di lasciarlo
> come metafora.

---

## 1. Dove il parallelismo regge, e dove no

Il parallelismo biologico mappa quattro livelli. L'architettura EMH reale
(vedi `osx-poc/src/eat/types.py::Tier`, README "EMH tiers") ne ha **tre**,
più i pesi statici del modello che non sono un tier EMH affatto:

| Proposta (biologica)   | EMH reale                                  | Nota di onestà |
|-------------------------|---------------------------------------------|----------------|
| Tier 0 "Adrenalina"     | **Non esiste come tier fisico separato.** VRAM è già il tier più veloce (`Tier.VRAM`, `types.py:26`). Non c'è un livello "registri/SRAM" indirizzabile separatamente da vLLM/CUDA in questo stack — sarebbe dentro l'attenzione del kernel stesso, fuori dal controllo di EMH. | Il concetto utile qui non è un *tier* in più, è una **classe di priorità di scheduling** dentro Tier.VRAM (vedi §3). |
| Tier 1 "Massima Attenzione" | `Tier.VRAM` — EMH-1a, RTX 3090 (`types.py:26`) | Mappa diretta. |
| Tier 2 "L'Arte Imparata"    | `Tier.DDR4` — EMH-1c, buffer caldo (`types.py:27`), con `Tier.NVME` — EMH-3 sotto (`types.py:28`) come oblio più profondo | La proposta comprime DDR4+NVMe in un solo "Tier 2"; nel codice reale sono due tier distinti con costi di transizione diversi (`TierManager._nvme_to_ddr4` vs `_ddr4_to_vram`, `manager.py:123-162`). |
| Tier 3 "Doti innate"       | I pesi del modello (checkpoint caricato da vLLM) | Non è gestito da EAT/TierManager — è fuori dal perimetro EMH per costruzione, non per omissione. |

**Correzione onesta rispetto alla discussione originale**: non serve un
quarto tier hardware. Serve un canale di **priorità/preemption** che intersechi
i tier esistenti, non un tier in più.

---

## 2. Cosa esiste già nel codice per due dei tre assi proposti

La discussione proponeva tre "vettori di stato": calore (frequenza), rilevanza
semantica, urgenza. **I primi due sono già implementati**, non da inventare:

`osx-poc/src/tier/policies.py::SEEPolicy.score()` (righe 64–100):

```python
score(shard) = α·freq_component + β·recency_component + γ·σ(shard, context)
# default: α=0.3, β=0.3, γ=0.4
```

- **Calore / frequenza** → `freq_component`, da `EATEntry.access_count`
  (`types.py:43`). Reale, testato.
- **Recency** → `recency_component`, da `EATEntry.last_access_ts`
  (`types.py:44`). Reale, testato.
- **Rilevanza semantica ("passaporto VIP")** → `σ(shard, context)`, il
  termine `γ`. **Oggi è uno stub**: `SEEPolicy.score()` fissa `sigma = 0.0`
  quando `context_vec` è fornito (`policies.py:97`), e ridistribuisce i pesi
  su α/β quando `context_vec` è `None` (`policies.py:88-92`). Il commento nel
  codice è esplicito: "σ è ancora uno stub (PT-PEP arriva in M3)"
  (`policies.py:9`). PT-PEP (`scheduler/ptpep.py`) esiste e produce
  `PTPEPPrediction.confidence`/`all_scores`, ma **non è collegato** a
  `SEEPolicy` — nessun caller passa oggi un `context_vec` reale da PT-PEP a
  `evict_to_free_vram()` (`manager.py:289-322`). Questo è un gap di wiring
  pre-esistente, non introdotto da questa proposta.

Quindi: "calore" e "rilevanza semantica" **non sono un modulo nuovo da
progettare** — sono `α`/`β` già in produzione e `γ` già previsto ma non
cablato. Il vero pezzo mancante, discusso sotto, è il terzo asse.

---

## 3. Il pezzo che manca davvero: urgenza come preemption, non come punteggio

L'"adrenalina" descritta nella discussione — bypassare la coda normale
quando succede qualcosa di critico — **non è un problema di scoring**. È un
problema di **priorità di esecuzione** su un sistema che oggi non ne ha
nessuna:

- `TierManager.promote()` (`manager.py:82-121`) serializza le transizioni
  per-key con un `asyncio.Lock` (`manager.py:60-78`), ma non ha alcuna nozione
  di priorità tra key diverse — due `promote()` concorrenti su shard diversi
  competono solo per l'event loop, FIFO de facto.
- `TierManager.prefetch()` (`manager.py:326-345`) lancia un batch di
  `promote()` con `asyncio.gather` — nessun modo di dire "questo shard salta
  la fila".
- Il precedente concettualmente più vicino nel codice reale è
  `GCSGGuard.should_activate_shadow()` (`scheduler/gcsg.py:237-266`): una
  cascata di soglie (`theta_gate`, `theta_entropy`, `theta_contamination`)
  che decide se bypassare il path normale (BF16) per un path shadow più
  economico. Non è preemption di scheduling, ma è lo stesso *stile* di
  decisione — soglie esplicite, cascata ordinata, motivo di skip loggato —
  che l'urgenza dovrebbe riusare, non reinventare.

### 3.1 Interfaccia proposta

Non tocca alcun file esistente. Nuovo modulo ipotetico
`osx-poc/src/scheduler/ecs.py` (nome di lavoro, da validare):

```python
@dataclass
class UrgencyContext:
    """Analogo a GatingContext (gcsg.py:117) ma per il segnale di urgenza.
    Prodotto PRIMA della tokenizzazione, come PT-PEP — stesso punto di
    aggancio, stesso motivo: serve prima che il forward pass parta, non
    durante.
    """
    request_id:   str
    urgency_score: float        # [0, 1] — da un classificatore leggero, TBD
    trigger_terms: List[str]    # termini che hanno alzato lo score, per debug/log
    timestamp:     float = field(default_factory=time.monotonic)


class UrgencyClassifier:
    """Analogo di scope a PTPEPClassifier (scheduler/ptpep.py) — CPU-only,
    nessuna dipendenza CUDA/torch a import time (stesso vincolo di PT-PEP,
    ptpep.py:16).

    DOMANDA APERTA, non risolta da questo documento: quale segnale alimenta
    urgency_score? Candidati, nessuno verificato:
      (a) pattern-matching su termini espliciti ("STOP", "ERRORE CRITICO",
          cambio lingua improvviso) — cheap, fragile, falsi negativi ovvi.
      (b) riuso del token_entropy già calcolato da GCSG (gcsg.py:129) come
          proxy di "sorpresa" — riusa un segnale che esiste già, ma
          token_entropy alta oggi significa "bassa confidenza del gating",
          non "urgenza semantica": sarebbe un riuso opportunistico da
          giustificare con dati, non per analogia.
      (c) un classificatore dedicato stile PT-PEP (TF-IDF+centroidi) su un
          dominio "urgenza" — richiede dataset etichettato che non esiste,
          stesso costo di training di PT-PEP (Sprint 3, vedi
          scripts/build_ptpep_classifier.py) senza garanzia che urgenza sia
          separabile linearmente come i domini semantici lo sono stati.
    Questo documento non sceglie tra (a)/(b)/(c) — è una decisione che
    richiede un dataset e una misurazione, non un'opinione.


class EmotionalCognitiveScheduler:
    """Estende SEEPolicy con un canale di priorità — non sostituisce la
    logica α/β/γ esistente, la precede.

    theta_urgency: soglia sopra cui una promote() ottiene preemption.
        Nessun default proposto qui — calibrarlo richiederebbe la stessa
        procedura empirica usata per GCSG (theta_gate=0.85 non è stato
        indovinato, vedi gcsg.py header) o PT-PEP
        (similarity_temperature=0.03, calibrato su
        tests/fixtures/ptpep_validation.json, ptpep.py:227-235).
    """

    def __init__(self, see_policy: SEEPolicy, theta_urgency: float) -> None:
        self._see = see_policy
        self.theta_urgency = theta_urgency

    def decide(
        self, entry: EATEntry, ctx: Optional[UrgencyContext],
        context_vec: Optional[list[float]] = None,
    ) -> "TierDecision":
        # 1. Adrenalina — preemption esplicita, PRIMA di ogni altro calcolo.
        #    Stesso pattern "return early col motivo" di
        #    GCSGGuard.should_activate_shadow (gcsg.py:249-266), non una
        #    metafora nuova.
        if ctx is not None and ctx.urgency_score >= self.theta_urgency:
            return TierDecision(
                action=Action.PROMOTE, target=Tier.VRAM,
                priority=Priority.PREEMPT,   # NUOVO — non esiste oggi in
                                              # TierManager.promote()/prefetch()
                reason=f"urgency_score {ctx.urgency_score:.3f} >= "
                       f"theta_urgency {self.theta_urgency}",
            )

        # 2. Passaporto VIP — non nuovo, è γ già previsto in SEEPolicy
        #    (policies.py:97), semplicemente MAI alimentato con un
        #    context_vec reale oggi. Wiring PT-PEP -> SEEPolicy, non nuova
        #    logica.
        see_score = self._see.score(entry, context_vec=context_vec)

        # 3. Fallthrough — comportamento SEE esistente, invariato.
        return TierDecision(action=Action.HOLD, target=entry.tier,
                             reason=f"see_score={see_score:.3f}")
```

### 3.2 Cosa richiederebbe in `TierManager` (non implementato, elenco onesto)

`priority=Priority.PREEMPT` sopra non fa nulla da solo — `TierManager`
(`manager.py`) dovrebbe:

1. Accettare una `priority` in `promote()` (oggi non ha quel parametro,
   `manager.py:82`).
2. Avere un modo per una `promote()` ad alta priorità di **non aspettare**
   dietro `asyncio.Lock` per-key già preso da una `promote()`/`evict()` a
   priorità normale sulla stessa key — oggi il lock è cieco alla priorità
   (`_lock_for`, `manager.py:72-78`); serve o una coda a priorità reale, o
   accettare che il preemption valga solo per key diverse (più semplice, ma
   allora "preemption" è impreciso — sarebbe solo "priorità nella coda del
   sistema", non interruzione di un trasferimento già in corso).
3. `evict_to_free_vram()` (`manager.py:289-322`) dovrebbe sapere liberare
   spazio *per* uno shard urgente specifico, non solo per un target di byte
   generico — oggi non lega l'eviction a un beneficiario.

Nessuno di questi tre punti è banale, e nessuno è stato misurato. Questo è
un elenco di "cosa cambierebbe", non un piano di implementazione.

---

## 4. Cosa NON fa questa proposta

- Non introduce un tier fisico "Tier 0" — vedi §1, la conclusione è che non
  serve.
- Non tocca alcun file in `osx-poc/src/`.
- Non sceglie un classificatore di urgenza (§3.1) — servono dati prima di
  una scelta difendibile, stesso principio che ha guidato la scelta di
  TF-IDF+centroidi su BERT-small per PT-PEP (Sprint 3: baseline misurabile
  prima, upgrade dopo se non basta — README, sezione PT-PEP).
- Non stima un impatto su latenza/throughput. Qualunque numero sarebbe
  inventato senza un'implementazione da profilare — lo stesso motivo per cui
  Sprint 2 non ha pubblicato un target di "shard promotion latency" prima di
  avere `bench_tier.py` da eseguire su hardware reale.

## 5. Collocazione nella roadmap

Non è Sprint 6 (Stockholm — telemetria passiva, README "Development
roadmap": osserva `GCSGGuard`/`AERManager`/`TierManager`/`EAT`, non decide
nulla). ECS *deciderebbe* transizioni di tier, quindi è un modulo con lo
stesso peso di M3 (Expert Scheduler), non un layer di osservabilità. Se
questa proposta viene portata avanti, andrebbe scoping come un nuovo modulo
(M5?) con il suo preflight di calibrazione — stesso standard richiesto a
GCSG (memory math verificata prima del wiring, `gcsg.py` header) e a
PT-PEP (hit rate misurato su held-out set prima di dichiararlo pronto,
`ptpep.py` header) — non aggiunto silenziosamente dentro M2/M3 esistenti.
