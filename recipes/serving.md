# Serving a ROCmFP4 quant

## Required environment

```bash
env LD_LIBRARY_PATH=/path/to/ROCmFPX/build/bin:/opt/rocm/lib \
    HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    GGML_HIP_ENABLE_UNIFIED_MEMORY=1 \
  /path/to/ROCmFPX/build/bin/llama-server ...
```

All three matter. `HSA_OVERRIDE_GFX_VERSION=11.5.1` is needed for gfx1151. Unified memory is what lets a ~100 GB model work at all on this hardware. Putting the fork's `build/bin` first on `LD_LIBRARY_PATH` avoids a soname clash if you also have a Vulkan llama.cpp build on the same machine — that clash produces confusing failures that look like model problems.

## Required flags

`-dio` — not optional on large models. See [gotchas](../docs/gotchas.md).

`--flash-attn on` and `--cache-type-k q8_0 --cache-type-v q8_0` for KV footprint.

## Per-model flags worth knowing

**Reasoning models:** `--reasoning-budget 512` caps the thinking so the answer actually lands inside your token budget. Without it a thinking-heavy model can return empty `content`.

**Vision models:** pass the projector with `--mmproj mmproj-<model>-f16.gguf`. Without it the model loads fine and simply cannot see images.

**Agent-markup models:** if a harness parses raw `<think>` / `<tool_call>` markup itself, use `--reasoning-format none` — llama.cpp's default reasoning extraction will otherwise rewrite that markup and break the parser downstream.

## Model swapping

On 128 GB unified memory, one ~100 GB model at a time. If you're running a model manager, put large quants in an exclusive single-slot group so loading one evicts the others rather than trying to coexist and failing in a confusing way.
