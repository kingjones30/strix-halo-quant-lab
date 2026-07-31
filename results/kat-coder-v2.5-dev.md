# KAT-Coder-V2.5-Dev 35B-A3B — ROCmFP4 STRIX_LEAN

**Verdict: discarded. Not published, not switched to.** ROCmFP4 actively hurt this model where it mattered.

| Property | Value |
|---|---|
| Architecture | `qwen35moe` — **hybrid linear + full attention** |
| Params | 35B total / ~3B active |
| Context | 262K |
| Baseline | `Q4_K_M` |

## Measured

| prompt_n | decode: ROCmFP4 | decode: Q4_K_M | Δ |
|---|---:|---:|---:|
| ~9K | — | — | **+12%** |
| ~37K | **25 tok/s** | **40 tok/s** | **−37%** |

Size saving was only −2.4 GiB.

## Read

This is the result that makes single-context benchmarking indefensible. Measured only at ~9K, this looks like a +12% win and would have shipped. At ~37K — which is the working range for a 262K-context coding model — it is 37% *slower*. For the actual workload, standard Q4_K_M is simply the better quant.

The build itself was clean: fork reused, converted from HF safetensors, text-only, no errors. It was the quant *fit* that failed, not the process. Lowest active-parameter count of the four builds and the only one with hybrid-linear attention.

## Why it's here

Negative results are the cheapest thing to publish and the most expensive thing to rediscover. Anyone considering ROCmFP4 for a small-active hybrid-attention MoE can now skip a day of work.
