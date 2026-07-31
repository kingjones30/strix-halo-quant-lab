# Laguna-S-2.1 118B-A8B — ROCmFP4 STRIX_LEAN

**Verdict: ship it.** The clearest win of the four builds. Published as
[kingjones777/Laguna-S-2.1-Q4_0_ROCMFP4_STRIX_LEAN-GGUF](https://huggingface.co/kingjones777/Laguna-S-2.1-Q4_0_ROCMFP4_STRIX_LEAN-GGUF).

| Property | Value |
|---|---|
| Base model | `poolside/Laguna-S-2.1` |
| Architecture | `laguna` |
| Params | 118B total / ~8B active |
| Baseline | `Q4_K_M` |

## Measured

| Quant | prompt_n | prefill tok/s | decode tok/s | size |
|---|---:|---:|---:|---:|
| Q4_K_M | 8116 | 445.6 | **17.36** | 71 GiB |
| **ROCmFP4 STRIX_LEAN** | 8116 | 381.8 | **28.23** | **58.3 GiB** |
| Q4_K_M | 32223 | 372.4 | **14.15** | 71 GiB |
| **ROCmFP4 STRIX_LEAN** | 32223 | 327.3 | **20.32** | **58.3 GiB** |

- Decode: **+62.6% @ ~8K**, **+43.6% @ ~32K**
- Size: **−18%** (58.3 vs 71 GiB), 4.26 BPW
- Prefill: **−14% / −12%** — the honest tradeoff

## Quality

Parity on all four shared probes: the bat-and-ball problem (both correct at $0.05), an `is_palindrome` implementation plan, `get_weather` tool calls for Paris and Tokyo (clean JSON from both), and reasoning-budget behaviour. No regression observed.

## Read

~8B active parameters through the FP4 FFN kernels, conventional attention. This is the profile ROCmFP4 was built for, and the gain is large enough that the prefill cost is worth paying for any decode-bound workload.
