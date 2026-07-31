# Leanstral-1.5-119B-A6B — ROCmFP4 STRIX_LEAN

**Verdict: size win only, no decode gain.** Published anyway with an honest table, because a first-of-its-kind `deepseek2` ROCmFP4 build is useful evidence even when the answer is "don't bother for speed":
[kingjones777/Leanstral-1.5-119B-A6B-Q4_0_ROCMFP4_STRIX_LEAN-GGUF](https://huggingface.co/kingjones777/Leanstral-1.5-119B-A6B-Q4_0_ROCMFP4_STRIX_LEAN-GGUF).
Q4_K_M was kept as the served quant — there was no performance reason to switch.

| Property | Value |
|---|---|
| Base model | `mistralai/Leanstral-1.5-119B-A6B` (Apache-2.0) |
| Architecture | `deepseek2` (DeepSeek-V3 MLA + MoE) |
| MoE | 128 routed experts / 4 active + 1 shared (~6.5B active) |
| Baseline | `Q4_K_M` (~68 GiB) |

## Measured

| prompt_n | metric | STRIX_LEAN | Q4_K_M | Δ |
|---|---|---:|---:|---:|
| 7616 | prefill tok/s | **449.9** | 439.2 | +2.4% |
| 7616 | decode tok/s | **37.41** | 36.85 | +1.5% |
| 23081 | prefill tok/s | **184.3** | 182.2 | +1.2% |
| 23081 | decode tok/s | 23.70 | **25.83** | **−8.2%** |
| — | size | **~59 GiB** | ~68 GiB | **−13%** |

Decode is flat at ~8K and ~8% slower at ~23K. Prefill was slightly *better*, which breaks the usual −12/−14% prefill tradeoff — worth noting but not worth much on its own.

## Read

ROCmFP4 loads and runs `deepseek2`/MLA correctly and produces coherent Lean 4 — the format works, it just doesn't move decode here. The plausible mechanism: MLA attention dominates the time, and with only 4 experts active there is comparatively little work flowing through the FP4 FFN kernels.

This build killed an earlier hypothesis of mine that a 128-expert MoE would be a strong FP4 candidate. Expert *count* turned out to be the wrong thing to look at; active parameter count is the number that tracks the result.

## Trap found here

Mistral pre-stacks the experts as a combined `gate_up` tensor. When `n_embd == 2 · n_ff` (both 4096 in this model) the split is dimensionally **ambiguous** — the converter can take the wrong half, and the model then **loads and decodes fine while emitting pure garbage**. It does not error. Compare against a known-good `Q4_K_M`'s tensor layout to establish the correct order. See [`docs/gotchas.md`](../docs/gotchas.md).
