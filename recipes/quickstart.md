# Quickstart — running a ROCmFP4 model from scratch

The complete path from a fresh machine to a served model, with the failure modes
that actually happen. If you only want the four commands, they're in the
[README](../README.md#quickstart--from-zero-to-a-running-model).

## What you need

- An AMD **gfx1151** part — Ryzen AI Max+ 395 (Strix Halo). Other RDNA3/RDNA4 GPUs
  will likely work with a different `GPU_TARGETS`, but nothing here is measured on them.
- **ROCm 7.x** installed and working. Verify before anything else:
  ```bash
  rocminfo | grep gfx        # should print gfx1151
  hipconfig -R               # should print your ROCm path
  ```
  If `rocminfo` doesn't list your GPU, stop here and fix ROCm first. Every
  confusing failure downstream traces back to this.
- Disk for the model. Sizes are in [MODELS.md](../MODELS.md) — they range from
  ~4 GiB to ~240 GiB.

## 1. Build the fork

Full build notes are in [`build-and-quantize.md`](build-and-quantize.md). The short version:

```bash
git clone https://github.com/charlie12345/ROCmFPX
cd ROCmFPX
HIPCXX=$(hipconfig -l)/clang HIP_PATH=$(hipconfig -R) cmake -B build \
  -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=OFF -DLLAMA_BUILD_WEBUI=OFF
cmake --build build -j 4
```

**Use `-j 4`.** A higher job count exhausts memory during compilation on this
hardware and fails the build in a way that looks unrelated to parallelism.

Confirm the binary understands the quant types before downloading tens of gigabytes:

```bash
build/bin/llama-quantize --help | grep ROCMFP4
```

Empty output means the build didn't produce what you need — fix that first.

## 2. Get a model

Pick one from [MODELS.md](../MODELS.md). Repos with several quants let you fetch a
single file rather than the whole repository:

```bash
pip install huggingface_hub

# One specific file
hf download kingjones777/Granite-4.1-30B-ROCmFP4-GGUF \
  granite-4.1-30b-Q4_0_ROCMFP4_COHERENT.gguf --local-dir ./models

# Or a whole repo (only if you want every variant)
hf download kingjones777/LFM2-8B-A1B-ROCmFPX-GGUF --local-dir ./models
```

**Which quant should you take?** In descending order of how often it's the right answer:

| Suffix | Bits | Take it when |
|---|---:|---|
| `Q4_0_ROCMFP4_COHERENT` | ~4.5 | Default. Fastest, smallest, token embeddings protected at Q6_K |
| `Q4_0_ROCMFP4_STRIX_LEAN` | ~4.4 | Slightly smaller, Q5_K embeddings |
| `Q8_0_ROCMFPX` | ~8.3 | You want maximum fidelity and have the memory |
| `*_AGENT` | 6–8 | Tool-calling fidelity matters. **Not a size win** — see below |

`Q6_0_ROCMFPX_AGENT` lands around 7.4 bits per weight and performs like the
8-bit, not like a 6-bit. It is a quality option, never a size option.

### Sharded models

Large repos ship split files (`-00001-of-00005.gguf`). Download **all** parts into
the same directory and point `--model` at part 00001 — the loader follows the chain.

## 3. Serve it

```bash
env LD_LIBRARY_PATH=/path/to/ROCmFPX/build/bin:/opt/rocm/lib \
    HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    GGML_HIP_ENABLE_UNIFIED_MEMORY=1 \
  /path/to/ROCmFPX/build/bin/llama-server \
    --model ./models/granite-4.1-30b-Q4_0_ROCMFP4_COHERENT.gguf \
    --host 0.0.0.0 --port 8080 --n-gpu-layers 999 \
    --flash-attn on -dio --jinja \
    --ctx-size 32768 --cache-type-k q8_0 --cache-type-v q8_0
```

All three environment variables matter:

- `HSA_OVERRIDE_GFX_VERSION=11.5.1` — required for gfx1151
- `GGML_HIP_ENABLE_UNIFIED_MEMORY=1` — what makes a ~100 GiB model possible at all
- `LD_LIBRARY_PATH` with the fork's `build/bin` **first** — prevents a soname clash
  with any other llama.cpp build on the machine. That clash produces failures that
  look like model corruption.

Add **`-fit off`** for anything large. `-fit default` combined with `-dio` and
unified memory on a 100+ GiB model can livelock the kernel driver in D-state,
which requires a power cycle to clear.

## 4. Verify it actually works

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain what a B-tree is and when you would not use one."}],"max_tokens":300}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
```

**Read the answer.** A bad conversion can load cleanly and decode at full speed
while emitting fluent nonsense. Throughput tells you nothing about correctness —
only output does. Ask two real questions before you trust it.

If you're testing a reasoning model and get empty `content`, that's not a
failure — the token budget went entirely to reasoning. Add `--reasoning-budget 512`.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `unknown model architecture` / `unsupported tensor type` | Wrong binary. You're on stock llama.cpp, not the fork |
| Loads, then emits fluent nonsense | Bad conversion, or a `gate_up` split picking the wrong half when `n_embd == 2*n_ff` |
| Gateway/client can't reach the server | `llama-server` binds `127.0.0.1` by default. Pass `--host 0.0.0.0` |
| Whole box wedges, GPU unresponsive, processes stuck in D | `-fit default` on a large model. Power cycle, then use `-fit off` |
| Build fails with unrelated-looking errors | `-j` too high. Use `-j 4` |
| Vision model can't see images | Missing `--mmproj mmproj-<model>-f16.gguf` |
| Tool calls arrive as text instead of structured calls | Harness isn't unwrapping the model's native tool-call markup. Try `--reasoning-format none` |

More in [`docs/gotchas.md`](../docs/gotchas.md).

## Next

- Serving flags per model type: [`serving.md`](serving.md)
- Quantizing your own: [`build-and-quantize.md`](build-and-quantize.md)
- Serving through an OpenAI-compatible front end: [`lemonade.md`](lemonade.md)
- Benchmarking properly: [`../bench/`](../bench/)
