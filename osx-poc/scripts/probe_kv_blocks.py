"""Probe veloce: quanti blocchi KV-cache risultano per una data combinazione
di max_model_len/max_num_seqs/gpu_memory_utilization, via GCSGWorker (pinning
shadow pool incluso) — senza eval completa, solo il numero. Issue #10/#16
punto 3.
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--max-model-len", type=int, default=3328)
parser.add_argument("--max-num-seqs", type=int, default=1)
parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
parser.add_argument("--quantization", choices=["awq", "awq_marlin"], default="awq_marlin")
args = parser.parse_args()

from vllm import LLM

MODEL_PATH = "/data/nvme/models/mixtral-instruct-awq"

print(f"max_model_len={args.max_model_len}, max_num_seqs={args.max_num_seqs}, "
      f"gpu_memory_utilization={args.gpu_memory_utilization}, quantization={args.quantization}")

try:
    llm = LLM(
        model=MODEL_PATH,
        worker_cls="scheduler.gcsg.GCSGWorker",
        quantization=args.quantization,
        cpu_offload_gb=4,
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
