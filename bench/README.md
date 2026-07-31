# Benchmark harness

`ab_bench.py` compares two or more GGUF quants of the same model on one machine.

```bash
cp example-config.json my-config.json     # edit paths
python3 ab_bench.py --config my-config.json --dry-run   # validate first
python3 ab_bench.py --config my-config.json
```

Requires Python 3.8+ and nothing else. It talks to `llama-server` over HTTP.

Output: `results.json` in `outdir`, plus a markdown table on stdout you can paste into a model card.

## What it controls for

**Equal generation length.** Every request sets `ignore_eos` and a fixed `max_tokens`, so every run generates exactly the same number of tokens. Decode tok/s measured over 4 tokens and over 256 tokens are not the same quantity, and comparing them produces confident nonsense.

**Prefix cache contamination.** Each prompt begins with a fresh UUID before any shared text, and `cache_prompt` is disabled. `cached_tokens` is recorded for every run and surfaced in the table with a ⚠ marker if it was ever non-zero. A warm prefix makes prefill look far better than it is — if you see that warning, the prefill column for that cell is invalid.

**Cold loads, one model at a time.** The harness kills any running `llama-server` before each model and reports load time separately. On unified-memory hardware two large models will not both fit, and a partially-swapped model produces meaningless numbers.

**Run-to-run noise.** Three runs per cell by default, median reported. Raw per-run values are all kept in `results.json` so you can see the spread rather than trusting a single number.

**Prompt length calibration.** Prompts are grown against the server's own `/tokenize` endpoint until they land within 2% of the target, so "~8K" means measured 8K rather than estimated 8K. The actual `prompt_tokens` is recorded per run.

## Timing source

If `llama-server` returns a `timings` object, its `prompt_per_second` and `predicted_per_second` are used — these separate prefill from decode properly. If it doesn't, the harness falls back to wall-clock decode (which includes prefill and so slightly understates decode) and marks the run `timings_source: wall_clock`. Check that field before quoting a number.

## Notes

`server_args` is passed through verbatim, so this works for any llama.cpp build, not just ROCmFP4 — drop the `env` block and the `-dio` flag for a stock build.

Test at more than one context length. A quant that wins at 8K can lose at 32K; that has happened here (see `results/kat-coder-v2.5-dev.md`).
