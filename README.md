# Strix Halo Quant Lab

Measured results from quantizing large models to **ROCmFP4** for AMD Ryzen AI Max+ 395 (Strix Halo, gfx1151, 128 GB unified memory), plus the benchmark harness used to produce them.

The ROCmFP4 format is not mine — that's the [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX) fork of llama.cpp. What's here is the part I could not find anywhere when I started: **which architectures actually get faster, by how much, and which ones don't.**

Every number below was measured on the same machine with the same binary. Failures are included.

## The short version

ROCmFP4 does **not** universally speed up decode. Across four builds, the models that gained were the ones with more active parameters flowing through the FP4 FFN kernels. Models using MLA attention or hybrid-linear attention with few active parameters gained nothing on decode, and one of them got materially worse at long context.

If you're deciding whether to spend a day quantizing something, that's the call you're making.

## Results

| Model | Arch | Active params | Decode @ short ctx | Decode @ long ctx | Size | Verdict |
|---|---|---:|---:|---:|---:|---|
| [Laguna-S-2.1 118B-A8B](https://huggingface.co/kingjones777/Laguna-S-2.1-Q4_0_ROCMFP4_STRIX_LEAN-GGUF) | `laguna` | ~8B | **+62.6%** | **+43.6%** | −18% | ship it |
| [Step-3.7-Flash 198B MoE](https://huggingface.co/kingjones777/Step-3.7-Flash-Q4_0_ROCMFP4_STRIX_LEAN-GGUF) | `step35` | ~11B | **+18%** | **+20%** | +10% | ship it |
| [Leanstral-1.5 119B-A6B](https://huggingface.co/kingjones777/Leanstral-1.5-119B-A6B-Q4_0_ROCMFP4_STRIX_LEAN-GGUF) | `deepseek2` (MLA) | ~6.5B | +1.5% | **−8.2%** | −13% | size only |
| KAT-Coder-V2.5-Dev 35B-A3B | `qwen35moe` (hybrid linear) | ~3B | +12% | **−37%** | −2.4 GiB | discarded |

**Baselines are not uniform, and that matters.** Laguna, Leanstral and KAT-Coder were measured against `Q4_K_M`. Step-3.7-Flash was measured against `UD-IQ4_XS`. That's why Step-3.7 shows a size *increase* — ROCmFP4 Strix Lean is ~4.26 BPW and IQ4_XS is ~3.9 BPW, so it's a higher-bit quant winning on speed while costing disk. Against Q4_K_M, ROCmFP4 has come out 13–18% smaller every time.

Full per-model detail, including raw run series and quality checks, is in [`results/`](results/).

## The working hypothesis

**Active parameter count through the FP4 FFN kernels appears to decide whether ROCmFP4 helps decode.** ~8B and ~11B active gained substantially. ~6.5B active with MLA attention was flat. ~3B active with hybrid-linear attention regressed at long context.

Attention type is confounded with active-param count in this sample, so I can't separate them cleanly — MLA and linear attention both shift work away from the FFN path where the FP4 kernels live, which is a plausible mechanism for the same observation. Four data points is a hypothesis, not a proof. I'd expect it to be tested rather than believed, and I'll update this table as more builds land.

One clear secondary finding: **when a decode gain exists, it can shrink or invert as context grows.** Benchmarking at a single short context length would have called KAT-Coder a win. It isn't one.

## Method

The harness is in [`bench/`](bench/). Four things it does that I'd consider non-negotiable for this kind of comparison:

**Equal generation length.** Every run generates exactly 256 tokens with `ignore_eos`. Decode tok/s across runs with different token counts isn't comparable, and one early run of mine produced soft numbers because a baseline emitted only 4 tokens before stopping.

**Nonce-prefixed prompts.** Every prompt starts with a fresh UUID so the server's prefix cache can't serve a previously-seen prefix and inflate prefill. The harness records `cached_tokens` on every run so you can verify it was 0. I caught a contaminated run this way mid-campaign; the numbers changed once it was fixed.

**Two context lengths, minimum.** ~8K and ~32K. See KAT-Coder above for why.

**One model resident at a time, cold-loaded, three runs, median reported.** On 128 GB unified memory two large models will not coexist, and a half-swapped model produces meaningless numbers.

## Hardware and runtime

All results measured on a Ryzen AI Max+ 395 (gfx1151 / Radeon 8060S), 128 GB unified memory, ROCm 7.2.4, using a HIP build of the ROCmFPX fork.

Build and runtime recipes — including the flags that are genuinely mandatory rather than merely recommended — are in [`recipes/`](recipes/). The operational traps that cost me real time are collected in [`docs/gotchas.md`](docs/gotchas.md); if you're doing this yourself, that file is probably the most useful thing in the repo.

## Models

Published quants live on HuggingFace under [kingjones777](https://huggingface.co/kingjones777). They require the ROCmFPX fork — `Q4_0_ROCMFP4_*` is not a stock llama.cpp quant type and these files will not load in upstream llama.cpp, Ollama, or LM Studio.

## License

MIT for everything in this repo. The quantized model weights are separately licensed under their respective base model licenses.
