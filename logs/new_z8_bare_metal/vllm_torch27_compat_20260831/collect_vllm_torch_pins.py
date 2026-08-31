#!/usr/bin/env python3
"""Estrae il pin `torch` dichiarato da ogni release vLLM su PyPI.

Fonte: https://pypi.org/pypi/vllm/{version}/json, campo `info.requires_dist`.
Prodotto: vllm_torch_pins.txt (accanto a questo file).

Richiede vllm_pypi_index.json (https://pypi.org/pypi/vllm/json) nella stessa
directory — archiviato accanto per rendere il risultato riproducibile anche
se PyPI cambia (le release passate sono immutabili, ma l'indice cresce).
"""
import json, urllib.request, re, datetime, pathlib

HERE = pathlib.Path(__file__).parent

def main() -> None:
    idx = json.load(open(HERE / "vllm_pypi_index.json"))
    rels = [v for v in idx["releases"] if idx["releases"][v]]
    def key(v):
        m = re.match(r"(\d+)\.(\d+)\.(\d+)", v)
        return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)
    rels = sorted(rels, key=key)
    print("# fonte: https://pypi.org/pypi/vllm/{version}/json  campo info.requires_dist")
    print("# comando: vedi collect_vllm_torch_pins.py in questa directory")
    print(f"# generato: {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}  host: berlin-3eie")
    print()
    print(f"{'vllm':16s} {'upload':12s} {'requires_python':16s} torch pin")
    for v in rels[rels.index("0.6.6.post1"):]:
        with urllib.request.urlopen(f"https://pypi.org/pypi/vllm/{v}/json", timeout=30) as r:
            m = json.load(r)
        t = [x for x in (m["info"].get("requires_dist") or [])
             if re.match(r"^torch\s*[=<>~]", x.strip())]
        up = idx["releases"][v][0]["upload_time"][:10]
        print(f"{v:16s} {up:12s} {m['info'].get('requires_python',''):16s} "
              f"{t[0] if t else 'NESSUN PIN'}")

if __name__ == "__main__":
    main()
