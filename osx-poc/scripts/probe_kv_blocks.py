"""Probe veloce: quanti blocchi KV-cache risultano per una data combinazione
di max_model_len/max_num_seqs/gpu_memory_utilization/cpu_offload_gb, via
GCSGWorker (pinning shadow pool incluso) — senza eval completa, solo il
numero. Issue #10/#16 punto 3.

Esteso 2026-08-12 (Sprint 4 sotto-obiettivo 6, issue #17) per il probe di
path 1 (`_ShadowExpertINT4`) sotto offload reale: --model-path e
--cpu-offload-gb ora configurabili (prima hardcoded a
mixtral-instruct-awq e cpu_offload_gb=4), --quantization accetta anche
"none" per il checkpoint fp16 non quantizzato che innesca path 1
(vedi scheduler.gcsg module docstring — FusedMoE con w13_weight grezzo,
non pre-quantizzato). Un checkpoint fp16 da ~93GB su una GPU da 24GB
richiede cpu_offload_gb nell'ordine di ~75-80GB, mai provato prima a
questa scala in questo progetto — questo probe serve esattamente a
trovare un valore che funziona PRIMA di lanciare uno smoke test/eval
completo che rischierebbe di bloccarsi a metà su un OOM o un budget
KV-cache negativo.
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", default="/data/nvme/models/mixtral-instruct-awq")
parser.add_argument("--max-model-len", type=int, default=3328)
parser.add_argument("--max-num-seqs", type=int, default=1)
parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
parser.add_argument("--quantization", choices=["awq", "awq_marlin", "none"], default="awq_marlin")
parser.add_argument("--cpu-offload-gb", type=float, default=4)
args = parser.parse_args()

from vllm import LLM

quantization = None if args.quantization == "none" else args.quantization

print(f"model_path={args.model_path}, max_model_len={args.max_model_len}, "
      f"max_num_seqs={args.max_num_seqs}, gpu_memory_utilization={args.gpu_memory_utilization}, "
      f"quantization={quantization}, cpu_offload_gb={args.cpu_offload_gb}")

try:
    llm = LLM(
        model=args.model_path,
        worker_cls="scheduler.gcsg.GCSGWorker",
        quantization=quantization,
        cpu_offload_gb=args.cpu_offload_gb,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        hf_overrides={"head_dim": 128},
    )
    cache_config = llm.llm_engine.cache_config
    blocks = cache_config.num_gpu_blocks
    block_size = cache_config.block_size
    print(f"RESULT: OK, num_gpu_blocks={blocks}, block_size={block_size}, "
          f"total_tokens_capacity={blocks * block_size}")
except ValueError as e:
    print(f"RESULT: INIT FAILED (vedi '# GPU blocks: N' nel log sopra per il "
          f"numero comunque) — {e}")
