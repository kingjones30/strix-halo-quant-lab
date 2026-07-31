# Build and quantize

## Build the fork

`Q4_0_ROCMFP4_*` is not a stock llama.cpp quant type. You need [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX):

```bash
git clone https://github.com/charlie12345/ROCmFPX
cd ROCmFPX
HIPCXX=$(hipconfig -l)/clang HIP_PATH=$(hipconfig -R) cmake -B build \
  -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=OFF -DLLAMA_BUILD_WEBUI=OFF
cmake --build build -j 4
```

`-j 4`, not more — see [gotchas](../docs/gotchas.md). `LLAMA_BUILD_WEBUI=OFF` avoids an asset-download build failure; Vulkan is off because this is a HIP-only build.

**Check your architecture is supported before you download anything.** Grep for `LLM_ARCH_<YOUR_ARCH>` in `src/llama-arch.h`, and confirm the quant type exists:

```bash
build/bin/llama-quantize --help | grep ROCMFP4
```

Note that arch names live in `build/bin/libllama.so`, not in the thin binaries — `strings build/bin/llama-quantize | grep <arch>` returns 0 even when the arch is fully supported. Use `libllama.so`, and sanity-check your method against a known-present arch first.

## Quantize

```bash
build/bin/llama-quantize \
  Model-F16.gguf \
  Model-Q4_0_ROCMFP4_STRIX_LEAN.gguf \
  Q4_0_ROCMFP4_STRIX_LEAN 8
```

Prefer an F16/BF16 source. If the model is too large to hold alongside the output, a `Q8_0` source with `--allow-requantize` works and measured clean for Step-3.7-Flash — but say so on the model card. People are entitled to know they're getting a requantization.

Available variants (`llama-quantize --help` for the full list): `STRIX_LEAN` (~4.38 BPW, Q5_K token embeddings), `STRIX` (~4.49), `FAST` (4.25, single-scale layout), `COHERENT` (4.70, Q6_K embeddings). `STRIX_LEAN` is what everything here was built with.

## Load-proof before deleting anything

```bash
env LD_LIBRARY_PATH=/path/to/ROCmFPX/build/bin:/opt/rocm/lib \
    HSA_OVERRIDE_GFX_VERSION=11.5.1 GGML_HIP_ENABLE_UNIFIED_MEMORY=1 \
  /path/to/ROCmFPX/build/bin/llama-server \
  --model Model-Q4_0_ROCMFP4_STRIX_LEAN.gguf \
  --host 127.0.0.1 --port 8080 --n-gpu-layers 999 \
  --flash-attn on -dio --no-warmup --jinja \
  --ctx-size 32768 --cache-type-k q8_0 --cache-type-v q8_0
```

Ask it two real questions and read the answers. A bad conversion can produce a model that loads and decodes at full speed while emitting fluent nonsense — only output tells you. Keep the source weights until it demonstrably talks sense.

## Then benchmark

See [`bench/`](../bench/). Two context lengths, equal generation counts, nonce-prefixed prompts.
