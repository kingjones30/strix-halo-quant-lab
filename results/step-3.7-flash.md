# Step-3.7-Flash 198B MoE — ROCmFP4 STRIX_LEAN

**Verdict: ship it.** First ROCmFP4 build of this model. Published as
[kingjones777/Step-3.7-Flash-Q4_0_ROCMFP4_STRIX_LEAN-GGUF](https://huggingface.co/kingjones777/Step-3.7-Flash-Q4_0_ROCMFP4_STRIX_LEAN-GGUF).

| Property | Value |
|---|---|
| Base model | `stepfun-ai/Step-3.7-Flash` (Apache-2.0) |
| Architecture | `step35` |
| Params | 198B total (196B LM + 1.8B vision) / ~11B active |
| Baseline | `unsloth/Step-3.7-Flash-GGUF` `UD-IQ4_XS` |
| Source | official `Q8_0` GGUF + `--allow-requantize` (BF16 is 394 GB, would not fit) |

## Measured

| | ROCmFP4 STRIX_LEAN | UD-IQ4_XS | Δ |
|---|---:|---:|---:|
| Prefill @ 8K | 331 tok/s | 296 tok/s | **+12%** |
| Decode @ 8K | 18.8 tok/s | 15.9 tok/s | **+18%** |
| Prefill @ 32K | 283 tok/s | 261 tok/s | **+8%** |
| Decode @ 32K | 14.9 tok/s | 12.4 tok/s | **+20%** |
| Size | 97.7 GiB | 88.8 GiB | +10% |

Raw series (all runs `cached_tokens=0`, 256 generated tokens):

| | prefill | decode |
|---|---|---|
| ROCmFP4 @8K | 331.94, 330.99, 319.50 | 18.76, 18.81, 20.25 |
| ROCmFP4 @32K | 284.11, 281.20, 282.96 | 14.94, 14.92, 14.94 |
| IQ4_XS @8K | 300.10, 295.89, 294.89 | 15.94, 15.95, 15.96 |
| IQ4_XS @32K | 261.45, 261.59, 260.85 | 12.42, 12.43, 12.41 |

**On the size column:** this is the one build where ROCmFP4 came out *larger*, because the baseline is a sub-4-bit IQ quant (~3.9 BPW) rather than Q4_K_M. At ~4.26 BPW this is a higher-bit quant winning on speed and paying 10% disk for it. Unlike a Q4_K_M comparison, that is not a free lunch.

**Unusually, prefill improved too.** Every other build here paid a 12–14% prefill penalty. Step-3.7 gained at both ends.

## Quality

Equivalent, not identical. Both quants produced a correct `is_palindrome` with the same `replace + lower + [::-1]` approach and passing assertions, and both answered the "all but 9 die" riddle correctly with 9. IQ4_XS was noticeably more verbose in its chain-of-thought and needed a higher token budget before its final answer appeared — a practical point in ROCmFP4's favour when serving.

## Read

~11B active, the largest of the four, and the strongest MoE result. Together with Laguna this is the evidence for the active-parameter hypothesis.
