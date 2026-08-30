# Strix Halo Quant Lab

**Running large language models locally on AMD Ryzen AI Max+ 395 (Strix Halo) with ROCmFP4 4-bit quantization — measured benchmarks, build recipes, and 118 ready-to-run GGUF models.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hardware: Ryzen AI Max+ 395](https://img.shields.io/badge/hardware-Ryzen%20AI%20Max%2B%20395-ED1C24)](https://www.amd.com/en/products/processors/laptop/ryzen/ai-max.html)
[![GPU: gfx1151](https://img.shields.io/badge/GPU-gfx1151%20%2F%20Radeon%208060S-000000)](https://rocm.docs.amd.com/)
[![Models on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20models-kingjones777-yellow)](https://huggingface.co/kingjones777)

This repository documents **which model architectures actually get faster when quantized to ROCmFP4 on AMD Strix Halo hardware, by how much, and which ones don't.** That was the thing I could not find anywhere when I started.

The ROCmFP4 format is not mine — it comes from the [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX) fork of llama.cpp. What's here is the measurement work, the build recipes, the architecture ports needed to make specific models load at all, and every published quantization.

Every number below was measured on the same machine with the same binary. **Failures are included.**

---

## Quickstart — from zero to a running model

Assumes Linux, ROCm 7.x, and a Ryzen AI Max+ 395 (or another gfx1151 part).

```bash
# 1. Build the ROCmFPX fork (this is what understands the FP4 tensor types)
git clone https://github.com/charlie12345/ROCmFPX && cd ROCmFPX
HIPCXX=$(hipconfig -l)/clang HIP_PATH=$(hipconfig -R) cmake -B build \
  -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=OFF -DLLAMA_BUILD_WEBUI=OFF
cmake --build build -j 4

# 2. Pull a model (this one is 15.2 GiB and runs at ~13 tok/s)
pip install huggingface_hub
hf download kingjones777/Granite-4.1-30B-ROCmFP4-GGUF \
  granite-4.1-30b-Q4_0_ROCMFP4_COHERENT.gguf --local-dir ./models

# 3. Serve it
env LD_LIBRARY_PATH=./build/bin:/opt/rocm/lib \
    HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    GGML_HIP_ENABLE_UNIFIED_MEMORY=1 \
  ./build/bin/llama-server \
    --model ./models/granite-4.1-30b-Q4_0_ROCMFP4_COHERENT.gguf \
    --host 0.0.0.0 --port 8080 --n-gpu-layers 999 \
    --flash-attn on -dio --jinja \
    --ctx-size 32768 --cache-type-k q8_0 --cache-type-v q8_0
```

Then `curl http://localhost:8080/v1/chat/completions` like any OpenAI-compatible endpoint.

`-j 4` is deliberate — a higher job count fails the build on this hardware. The three environment variables are all load-bearing, `-dio` is not optional on large models, and `--host 0.0.0.0` matters if anything other than localhost needs to reach it. All of it is explained in [`recipes/quickstart.md`](recipes/quickstart.md), [`recipes/serving.md`](recipes/serving.md) and [`docs/gotchas.md`](docs/gotchas.md).

## Will these run on my hardware?

| Your setup | Will it work? |
|---|---|
| AMD Ryzen AI Max+ 395 / Strix Halo (gfx1151), ROCm | **Yes** — this is the target |
| Other AMD ROCm GPU (RDNA3/RDNA4, gfx11xx/gfx12xx) | **Probably** — built for gfx1151; retarget `AMDGPU_TARGETS` and re-benchmark |
| Upstream llama.cpp, Ollama, LM Studio, Jan, koboldcpp | **No** — see below |
| NVIDIA (vLLM / SGLang) | Use the **NVFP4** repos in [MODELS.md](MODELS.md) instead |

### Why won't these load in Ollama or LM Studio?

Because `Q4_0_ROCMFP4` is a genuinely new tensor type, not a relabeled standard one. In the fork's `ggml.h` it occupies enum value **100**; upstream llama.cpp's tensor-type enum ends in the 30s. A stock runtime reads tensor type 100, finds it past its own `GGML_TYPE_COUNT`, and rejects the file.

Concretely: in the Granite-4.1-30b 4-bit build, **448 of 578 tensors** are type `Q4_0_ROCMFP4`. There is no metadata edit or repo tag that makes a stock runtime read them. You need the fork.

**Lemonade Server is the exception, and it is verified working.** AMD's own local server can be pointed at your own engine (`lemonade config set llamacpp.rocm_bin=/path/to/rocmfpx/llama-server`), so ROCmFP4 models serve through a standard OpenAI-compatible API. Measured 2026-08-17 on Lemonade 10.5.1: **98.5–101.6 tok/s** end-to-end for Ling-3.0-tiny FP4, against 105.2 tok/s for the same model on bare `llama-server` — the wrapper costs almost nothing. Full recipe, including two Lemonade path bugs whose error messages look like quantization failures: [`recipes/lemonade.md`](recipes/lemonade.md).

## The short version

ROCmFP4 does **not** universally speed up decode. Across the four fully A/B-tested builds, the models that gained were the ones with more active parameters flowing through the FP4 FFN kernels. Models using MLA attention or hybrid-linear attention with few active parameters gained nothing on decode, and one of them got materially worse at long context.

If you're deciding whether to spend a day quantizing something, that's the call you're making.

## A/B results — ROCmFP4 vs the standard quant

| Model | Arch | Active params | Decode @ short ctx | Decode @ long ctx | Size | Verdict |
|---|---|---:|---:|---:|---:|---|
| [Laguna-S-2.1 118B-A8B](https://huggingface.co/kingjones777/Laguna-S-2.1-ROCmFP4-STRIX_LEAN-GGUF) | `laguna` | ~8B | **+62.6%** | **+43.6%** | −18% | ship it |
| [Step-3.7-Flash 198B MoE](https://huggingface.co/kingjones777/Step-3.7-Flash-ROCmFP4-STRIX_LEAN-GGUF) | `step35` | ~11B | **+18%** | **+20%** | +10% | ship it |
| [Leanstral-1.5 119B-A6B](https://huggingface.co/kingjones777/Leanstral-1.5-119B-A6B-ROCmFP4-STRIX_LEAN-GGUF) | `deepseek2` (MLA) | ~6.5B | +1.5% | **−8.2%** | −13% | size only |
| KAT-Coder-V2.5-Dev 35B-A3B | `qwen35moe` (hybrid linear) | ~3B | +12% | **−37%** | −2.4 GiB | discarded |

**Baselines are not uniform, and that matters.** Laguna, Leanstral and KAT-Coder were measured against `Q4_K_M`. Step-3.7-Flash was measured against `UD-IQ4_XS`. That's why Step-3.7 shows a size *increase* — ROCmFP4 Strix Lean is ~4.26 BPW and IQ4_XS is ~3.9 BPW, so it's a higher-bit quant winning on speed while costing disk. Against Q4_K_M, ROCmFP4 has come out 13–18% smaller every time.

Full per-model detail, including raw run series and quality checks, is in [`results/`](results/).

## Measured throughput on Ryzen AI Max+ 395

Single-stream decode, 128 GB unified memory, ROCm 7.2.4, median of 3 runs. These are absolute numbers for the 4-bit builds, not A/B deltas.

| Model | Decode | Notes |
|---|---:|---|
| LFM2-8B-A1B | **146.6 tok/s** | MoE, 1B active |
| Ling-3.0-tiny | **97.6 tok/s** | required an architecture port (below) |
| Mellum2-12B-A2.5B | **96.9 tok/s** | |
| LFM2-24B-A2B | **95.2 tok/s** | |
| Gemma-4-E2B | **80.9 tok/s** | MatFormer |
| Gemma-4-E4B | **49.9 tok/s** | MatFormer |
| Muse-Glimmer-30B | **15–39 tok/s** | with DFlash drafter; see the range caveat below |
| Qwen3.8-27B | **30.3 tok/s** | with MTP draft head = **2.83×** over no drafter |
| Llama-4-Scout-17B-16E | **17.7 tok/s** | vision |
| Granite-4.1-30b | **13.1 tok/s** | dense, 30B |

**Speculative-decode throughput is workload-dependent — treat a single number as wrong.** Muse-Glimmer measures 15 tok/s on prose and 39 tok/s on code with the same binary and the same drafter, because draft acceptance depends on how predictable the output is. Quote a range or state the workload.

**Always benchmark with the drafter the repo ships.** Benchmarking Qwen3.8-27B without its MTP head gives 13.8 tok/s instead of 38.3 — a number that is not wrong so much as measuring a configuration nobody runs.

## Checking that a benchmark is actually complete

Weights stream once per token, so `tok/s × file_GB` gives effective memory bandwidth. Strix Halo's peak is ~256 GB/s. On all four Granite-4.1-30b quants:

| Quant | GB/s | % of peak |
|---|---:|---:|
| 4-bit | 213.7 | **83.5%** |
| 6-bit AGENT | 207.3 | 81.0% |
| 8-bit | 210.0 | 82.0% |
| 8-bit AGENT | 208.0 | 81.2% |

Tightly clustered at the hardware ceiling means the model is purely bandwidth-bound and there is nothing left to tune. **"Is decode ≥ ~80% of peak bandwidth?" is a positive test that catches every form of under-benchmarking at once** — a missing draft head, CPU-spilled layers, `--fit` renegotiation, another process contending for the GPU. Well below 80% on a dense model means investigate before publishing.

**⚠️ This check is only valid for conventional dense models.** It assumes every byte of the file is read once per token. Two families break that assumption:

- **MoE** — only active experts are read. Compute against *active* weight, never file size.
- **MatFormer / per-layer-embedding** (Gemma-4 E2B/E4B, Gemma-3n) — a large share of parameters sits in `per_layer_token_embd` and is not read per token.

On Gemma-4-E2B the arithmetic returns 98.8% and 102% of "peak" — *above* the ceiling, which is physically impossible for a full per-token read. A result over 100% is the tell that your denominator is wrong, not evidence of a great build.

## Architecture ports

Several models could not be quantized or loaded at all until the loader was fixed. These were the interesting problems:

- **`bailingmoe3` (Ling-3.0-tiny)** — the in-tree implementation assumed `q_lora_rank=null`, which is true for Ling-3.0-flash but not for tiny, where it is 256. No GGUF of Ling-3.0-tiny could load, from anyone, until that was handled. **When the BF16 source fails the same way your quant does, the bug is in the loader, not the artifact.**
- **Speculative decoding disabled prompt caching** — with an MTP draft head attached, the cache never hit, so every agentic turn reprocessed the full prompt: `prompt_n 8045`, `prompt_ms 27948`. Capturing the speculative state boundary inside the checkpoint fixed it to `prompt_n 4`, `prompt_ms 100`, gated byte-identical across 10 runs.
- **`Gemma4Assistant requires ctx_other`** — Gemma-4's shipped MTP drafter runs at `n_embd=256` against a 1536/2560-wide target and cannot currently be served. Documented rather than papered over; those speeds are not advertised.

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

## Published models

**118 repositories** — 108 ROCmFP4/ROCmFPX for AMD Strix Halo, 9 NVFP4 for NVIDIA, 1 other. **126,066 downloads** in the last 30 days.

**→ [Full index with sizes, base models and contents: `MODELS.md`](MODELS.md)**

That file is generated directly from the Hugging Face API by [`tools/gen_model_index.py`](tools/gen_model_index.py), so it cannot drift from what is actually published:

```bash
python3 tools/gen_model_index.py > MODELS.md
```

Highlights, all first published builds of their kind in this format:

- **Qwen3.8-Flash-Next** — the first `qwen4exp` GGUFs published anywhere; 5 repos across FAST / STRIX /
  STRIX_LEAN and uncensored variants, 131K context proven on a 128 GB box
- **Ornith-1.5 35B-A3B and 9B** — full ROCmFP4 ladders plus abliterated and NVFP4-vision variants.
  On the 35B, **Vulkan is +8.6–14.4% on decode while ROCm is +45–67% on prefill** — the backend is worth
  ~10× the tier. That ranking **does not transfer** to the dense 9B, where the two backends are within ~1%.
- **Granite-4.2 3B / 8B / 30B** — 4 tiers each (COHERENT, STRIX_LEAN, Q8_0, Q8_0-AGENT)
- **LFM2.5 1.2B / 2.6B / 8B-A1B** — ROCmFP4 + DSpark drafters + NVFP4, 11 repos
- **Ling-3.0 flash / tiny** — base, midtrain and 30T training stages published separately
- **Granite-4.1 3B / 8B / 30B** — dense; the 30B runs 13.1 tok/s at 83.5% of peak bandwidth
- **Gemma-4 E2B / E4B / 12B / 31B / 26B-A4B** — MatFormer, vision, mmproj shipped
- **LFM2-8B-A1B / LFM2-24B-A2B** — 146.6 / 95.2 tok/s
- **Tiel-Coder-35B-A3B**, **Apodex-1.1-mini**, **DiffusionGemma-26B-A4B**, **Qwen3-VL-8B**
- **DeepSeek-V4-Flash-180B** — 181.7 GiB
- **Llama-4-Scout-17B-16E** — 57 GiB, vision verified

## Hardware and runtime

All results measured on a Ryzen AI Max+ 395 (gfx1151 / Radeon 8060S), 128 GB unified memory, ROCm 7.2.4, using a HIP build of the ROCmFPX fork.

Build and runtime recipes — including the flags that are genuinely mandatory rather than merely recommended — are in [`recipes/`](recipes/). The operational traps that cost me real time are collected in [`docs/gotchas.md`](docs/gotchas.md); if you're doing this yourself, that file is probably the most useful thing in the repo.

## FAQ

### What is ROCmFP4?

A 4-bit quantization format using packed AMD FP4 blocks with UE4M3 scales, implemented in the ROCmFPX fork of llama.cpp. It targets AMD hardware directly rather than being a generic 4-bit scheme, which is why it can beat `Q4_K_M` on decode for some architectures — and why it does nothing for others.

### What is the difference between ROCmFP4 and ROCmFPX?

ROCmFP4 is the 4-bit type. ROCmFPX is the wider family — 2, 3, 6 and 8-bit variants sharing the same UE4M3-scale approach. Repos are named for what they contain.

### What do COHERENT, LEAN, STRIX and AGENT mean in the filenames?

They are quantization recipes that differ in which tensors get protected at higher precision. `COHERENT` keeps token embeddings at Q6_K; `LEAN` uses Q5_K; `AGENT` promotes a large share of tensors to 8-bit for tool-use fidelity. **`Q6_0_ROCMFPX_AGENT` is an 8-bit-class build, not a middle rung** — it lands ~7.4 bits per weight and performs like the 8-bit, so choose it for quality, never for size.

### How much RAM do I need?

The model file size plus KV cache. 128 GB unified memory comfortably runs models up to ~100 GiB one at a time. Two large models will not coexist. Use `--cache-type-k q8_0 --cache-type-v q8_0` to cut the KV footprint.

### Can I run these on the NPU?

No. These are GPU (iGPU) builds. The XDNA2 NPU on Strix Halo is a separate engine reached through FastFlowLM or Lemonade's hybrid path, and its decode throughput is lower than the iGPU's — its value is efficiency and being a second independent compute engine, not speed.

### Which model should I start with?

If you want speed, `LFM2-8B-A1B` at 146 tok/s. If you want a capable general model, `Qwen3.8-27B` with its MTP drafter. If you want to see what a dense 30B does on this hardware, `Granite-4.1-30b`.

## License

MIT for everything in this repo. The quantized model weights are separately licensed under their respective base model licenses.
