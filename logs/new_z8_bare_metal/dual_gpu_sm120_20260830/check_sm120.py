#!/usr/bin/env python3
"""Verifica sm_120 / RTX 5060 Ti su un dato stack torch — issue #8.

I tre controlli della checklist "Da fare prima/durante l'arrivo" dell'issue:
  1. device enumerati e VRAM (lato torch; nvidia-smi e' catturato a parte)
  2. torch.cuda.get_arch_list() -> sm_120 presente?
  3. operazione CUDA minima su ogni device -> "no kernel image" o OK?

Eseguito identico su due stack per avere il controfattuale:
  torch==2.5.1+cu124 (pin attuale, requirements-vllm.txt) -> atteso FALLIRE su cuda:1
  torch==2.7.0+cu128 (candidato)                          -> atteso PASSARE su cuda:1
"""
import torch

print(f"torch          : {torch.__version__}")
print(f"CUDA build     : {torch.version.cuda}")
print(f"device_count   : {torch.cuda.device_count()}")
print(f"arch_list      : {torch.cuda.get_arch_list()}")
print(f"sm_120 in list : {'sm_120' in torch.cuda.get_arch_list()}")
print(f"capabilities   : {[torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]}")
print()
print("--- operazione CUDA minima: torch.randn(10, device=cuda:i).sum() ---")
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    total = torch.cuda.get_device_properties(i).total_memory // (1024 ** 2)
    try:
        r = torch.randn(10, device=f"cuda:{i}").sum()
        print(f"cuda:{i}  {name}  ({total} MiB)  -> OK  sum={r.item()}")
    except Exception as e:
        print(f"cuda:{i}  {name}  ({total} MiB)  -> {type(e).__name__}: {e}")
