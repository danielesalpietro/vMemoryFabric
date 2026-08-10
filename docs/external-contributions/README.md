# Contributi esterni da valutare

Raccolta di analisi su progetti open source esterni a OSX, valutati per
possibili contributi (di design o di codice) alle sprint del roadmap
(`osx-poc/README.md` — Sprint 0–5, moduli M1–M4).

Ogni progetto valutato ha un proprio file `<progetto>.md` in questa cartella.
Non è un elenco di dipendenze da adottare: è un log di cosa è stato guardato,
cosa è risultato rilevante (e per quale sprint/modulo), cosa no, e perché.

## Indice

| Progetto | Licenza | Modulo/sprint più rilevante | File |
|---|---|---|---|
| [exo-explore/exo](https://github.com/exo-explore/exo) | Apache-2.0 | M2 (Sprint 2, Eketorp) — async shard downloader; M3/AER (stub) — placement topology-aware | [exo-explore-exo.md](exo-explore-exo.md) |

## Template per nuove voci

Quando si aggiunge un nuovo progetto, seguire questa struttura (vedi
`exo-explore-exo.md` come esempio):

1. **Cos'è** — una riga, cosa fa il progetto e perché è stato guardato.
2. **Licenza e compatibilità** — licenza del progetto vs. licenza OSX (MIT),
   eventuali vincoli.
3. **Rilevante da vicino** — cosa si applica allo sprint corrente o al
   prossimo, con link a file/moduli specifici del progetto esterno e al
   modulo OSX corrispondente.
4. **Rilevante a medio termine** — cosa si applica a moduli ancora stub o a
   sprint future.
5. **Non rilevante** — cosa è stato escluso e perché (stack incompatibile,
   problema diverso, ecc.), per non doverlo rivalutare da zero in futuro.
6. **In sintesi** — 2-3 righe con l'azione concreta consigliata (leggere un
   file come riferimento di design, aprire un issue, non fare nulla per ora).

Aggiornare anche la tabella dell'indice qui sopra.
