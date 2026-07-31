# Gotchas

Things that cost me real time. Roughly in order of how much.

## `-dio` is mandatory on large unified-memory loads

Without direct I/O the mmap path climbs to ~120 GB RSS with GTT stuck near 0 and sits there looking hung. It is not hung, and the file is not corrupt — it is the loader eagerly prepopulating the mapping before GPU placement. With `-dio` a 97.7 GiB artifact cold-loads in about 60 seconds.

This is the single most common false "the quant is broken" report. If a big ROCmFP4 model appears to hang on load, add `-dio` before you suspect anything else.

The DeepSeek-V4 path in the fork hits the same wall from a different angle and documents `--no-mmap` as its fix — a bounded syscall trace there showed `mmap(..., ~96 GB, MAP_SHARED|MAP_POPULATE)`. Same root cause, two workarounds.

## Never run a heavy compile and a heavy download at once

On 128 GB unified memory, a HIP compile concurrent with a large model download hard-OOMs the machine. Not a slowdown — it goes off the wire and needs a physical power cycle. Keep every heavy phase strictly sequential, and throttle the build to `-j 4`; the HIP flash-attention kernels spike RAM badly at `-j 8` and above.

## Read GTT from `card0` specifically

`/sys/class/drm/card0/device/mem_info_gtt_used`. A `card*` glob will pick up `card1`, which reports 0, and hand you a false "memory is clear" while ~112 GB is still pinned. This hid a real problem from me for a while.

## Reasoning models return empty `content` if the budget is too low

Step-3.7-Flash will spend an entire `max_tokens` on chain-of-thought and hand back an empty `content` with `finish_reason: length`. It looks exactly like a broken quant. It isn't — cap the thinking with `--reasoning-budget 512` (or raise `max_tokens`) and the answer lands.

Worth knowing generally: when comparing two quants of a reasoning model, an under-budgeted baseline will look like it produced a wrong answer when it simply never got to the answer. Give both sides enough room.

## The `gate_up` split can silently produce a garbage model

Some publishers pre-stack MoE experts into a combined `gate_up` tensor. When `n_embd == 2 · n_ff`, the split is dimensionally ambiguous — the converter can take the wrong half, and the resulting model **loads, decodes at full speed, and emits fluent garbage**. There is no error anywhere.

Hit on Leanstral-1.5-119B-A6B, where both dimensions are 4096. Fix is to flip the gate/up branch in the converter; verify by comparing tensor layout against a known-good `Q4_K_M` of the same model.

The general lesson is the important one: **load-proof for coherence before deleting your source weights.** "Conversion completed" and "quantization completed" tell you nothing about correctness. I nearly ate a 238 GB re-download learning this.

## Prefix caching will quietly fake your prefill numbers

If your benchmark prompts share a prefix, the server serves it from cache and prefill tok/s becomes meaningless. Put a fresh UUID at the **start** of every prompt and check `cached_tokens == 0` on every run. I caught a contaminated benchmark campaign mid-run this way; the numbers moved once it was fixed.

## Unequal generation lengths make decode tok/s incomparable

If one quant stops at 4 tokens and the other runs to 256, their decode rates are not the same measurement. Force `ignore_eos` with a fixed `max_tokens` on both sides. An early Leanstral comparison of mine produced soft numbers exactly this way.

## Benchmark at more than one context length

A quant that wins at 8K can lose badly at 32K. KAT-Coder-V2.5-Dev was +12% at ~9K and −37% at ~37K. Single-context benchmarking would have shipped a regression.

## Keep the machine dedicated through the upload, not just the build

Restoring your model-serving stack before a large HuggingFace upload finishes will starve it. A warmup timer reloading resident models drove a 59 GB upload down to ~1.2 MB/s and eventually killed it; on a freed machine the same upload ran at ~95 MB/s and finished in about ten minutes. Uploads are a large disk-read plus network job and need RAM headroom too.

Run long uploads and downloads detached with a logfile. A multi-tens-of-GB foreground transfer dies on any SSH blip.

## HuggingFace caps files at 50 GB

Anything larger has to be sharded. `llama-gguf-split --split --split-max-size 45G in.gguf out-prefix` produces the conventional `-00001-of-0000N.gguf` chain; point llama.cpp at the first shard and it picks up the rest. Load-proof the **shard chain**, not the original file — you're publishing the shards.
