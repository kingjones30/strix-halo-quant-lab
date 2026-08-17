# Serving ROCmFP4 models through Lemonade

> **Status: documented from AMD's official docs, not yet verified on this hardware.**
> Everything else in this repository is measured. This file is not — it is the
> vendor-documented path written down so it can be tested. If you try it before I
> do, please open an issue with the result either way.

## Why this is worth doing

ROCmFP4 files will not load in Ollama, LM Studio, Jan, or koboldcpp, because they
all bundle upstream llama.cpp and `Q4_0_ROCMFP4` is not an upstream tensor type
(see [the README](../README.md#why-wont-these-load-in-ollama-or-lm-studio)).

[Lemonade](https://github.com/lemonade-sdk/lemonade) is AMD's own open-source local
LLM server, and it is the one front end that has a documented way out of this: it
lets you point it at **your own backend binary** instead of the one it downloads.
That means a standard OpenAI-compatible server, model management, and a GUI — in
front of the ROCmFPX fork.

## The documented path

### 1. Install Lemonade

```bash
sudo add-apt-repository ppa:lemonade-team/stable
sudo apt update
sudo apt install lemonade-server
lemonade status
```

### 2. Point it at the ROCmFPX build

This is the load-bearing step. Per AMD's
[llama.cpp backend docs](https://lemonade-server.ai/docs/guide/configuration/llamacpp/),
you can supply your own binaries and Lemonade will use them instead of fetching
its own:

```bash
lemonade config set llamacpp.rocm_bin /path/to/ROCmFPX/build/bin
```

The fork's `build/bin` must contain `llama-server`. Environment-variable
equivalents exist if you'd rather not persist the config.

### 3. Register a model

Lemonade can register any Hugging Face GGUF repo, by exact filename or by
quantization shorthand:

```bash
lemonade pull kingjones777/Granite-4.1-30B-ROCmFP4-GGUF:granite-4.1-30b-Q4_0_ROCMFP4_COHERENT.gguf
```

This writes an entry to `user_models.json` in the Lemonade cache
(default `~/.cache/lemonade`). Alternatively, drop `.gguf` files into a directory
and set `extra_models_dir` — anything found there appears under the `custom` category.

### 4. Serve

```bash
lemonade run user.Granite-4.1-30B-ROCmFP4-GGUF
```

The endpoint is OpenAI-compatible at `http://localhost:13305/api/v1`.

## What is unverified, specifically

1. **Whether Lemonade's model-loading path passes the custom binary the flags it
   needs.** ROCmFP4 on large models wants `-dio`, `-fit off`, and the
   `HSA_OVERRIDE_GFX_VERSION` / `GGML_HIP_ENABLE_UNIFIED_MEMORY` environment. If
   Lemonade constructs its own launch line, those may not be reachable.
2. **Whether Lemonade validates quant types before handing off.** If it inspects
   the GGUF header itself and rejects unknown types, the custom binary never gets
   a chance.
3. **Whether the NPU/hybrid paths interact badly.** They should be irrelevant for
   a GGUF iGPU workload, but that is an assumption.

## Getting these models into Lemonade's built-in list

Unlike LM Studio, Lemonade has a real submission process. Per their
[FAQ](https://github.com/lemonade-sdk/lemonade/blob/main/docs/guide/faq.md), you can
open a pull request adding the model to the built-in `server_models.json`, or open
an issue requesting support. Maintainers review for device compatibility.

That is worth doing **after** the path above is verified end-to-end, not before.

## Notes

Lemonade's ROCm stable channel already ships custom llama.cpp builds from
`lemonade-sdk/llama.cpp` — AMD maintains its own patched llama.cpp for AMD
hardware. That makes it a more plausible home for FP4 tensor-type support than
upstream ggml-org, and it is the same organization behind the
[AMD AI Developer Program](https://www.amd.com/en/developer/ai-dev-program.html)
and [amd/playbooks](https://github.com/amd/playbooks).
