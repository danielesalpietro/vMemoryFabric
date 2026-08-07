"""Benchmark M1 — EAT latenza e throughput.

Da eseguire DOPO Sprint 1 (implementazione completata).
In dev, questi script restituiscono NotImplementedError — è normale.

Output: JSON con P50/P95/P99 latenza lookup, throughput insert, ecc.
"""
import json, sys, time
from pathlib import Path

def main():
    print("[bench_eat] TODO Sprint 1 — EAT non ancora implementata.")
    print("            Eseguire dopo implementazione in src/eat/.")
    # placeholder: output JSON vuoto per non rompere CI
    result = {"status": "pending", "sprint": 1, "module": "EAT"}
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
