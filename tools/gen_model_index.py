#!/usr/bin/env python3
"""Regenerate MODELS.md from the live Hugging Face API.

Every number in the generated file comes from the API at run time. Nothing is
hardcoded and nothing is estimated, so the index cannot drift from what is
actually published. Re-run it after any upload:

    python3 tools/gen_model_index.py > MODELS.md

Dependency-free (stdlib only), same as bench/ab_bench.py.
"""

import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

AUTHOR = "kingjones777"
API = "https://huggingface.co/api/models"
GIB = 1024 ** 3


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "strix-halo-quant-lab-index"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def list_models():
    out, page = [], 0
    while True:
        batch = get(f"{API}?author={AUTHOR}&limit=100&skip={page * 100}&full=true")
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return out


def detail(model_id):
    """Per-repo call: siblings carry real blob sizes, which the list call omits."""
    try:
        return get(f"{API}/{model_id}?blobs=true")
    except urllib.error.HTTPError as e:
        print(f"warn: {model_id} -> HTTP {e.code}", file=sys.stderr)
        return None


def classify(name, tags):
    """Bucket a repo by the runtime required to load it."""
    if "ROCmFP" in name or "rocmfp4" in tags or "rocmfpx" in tags:
        return "rocm"
    if "NVFP4" in name:
        return "nvfp4"
    return "other"


def describe(siblings):
    """Summarize repo contents from the actual file list."""
    ggufs = [s for s in siblings if s["rfilename"].lower().endswith(".gguf")]
    names = [s["rfilename"].lower() for s in ggufs]
    weights = [n for n in names if "mmproj" not in n]

    parts = []
    # Shard chains count as one model, not N variants.
    shard_stems = {n.split("-0000")[0] for n in weights if "-0000" in n}
    singles = [n for n in weights if "-0000" not in n]
    variant_count = len(shard_stems) + len(singles)

    if variant_count > 1:
        parts.append(f"{variant_count} quant variants")
    elif variant_count == 1:
        parts.append("single model")

    if any("mmproj" in n for n in names):
        parts.append("vision projector")
    if any(k in n for n in names for k in ("draft", "dflash", "mtp", "eagle")):
        parts.append("speculative drafter")

    return ", ".join(parts) if parts else "—"


def size_gib(siblings):
    total = sum(s.get("size") or 0 for s in siblings)
    return total / GIB if total else None


def base_link(card):
    bm = (card or {}).get("base_model")
    if isinstance(bm, list):
        bm = bm[0] if bm else None
    if not bm:
        return "—"
    return f"[`{bm}`](https://huggingface.co/{bm})"


def row(m):
    name = m["modelId"].split("/")[1]
    sz = size_gib(m.get("siblings") or [])
    sz_txt = f"{sz:.1f} GiB" if sz else "—"
    return (
        f"| [`{name}`](https://huggingface.co/{m['modelId']}) "
        f"| {base_link(m.get('cardData'))} "
        f"| {sz_txt} "
        f"| {describe(m.get('siblings') or [])} "
        f"| {m.get('downloads', 0):,} |"
    )


HEADER = "| Model | Base model | Size | Contents | Downloads (30d) |\n|---|---|---:|---|---:|"


def main():
    models = list_models()
    print(f"fetching file lists for {len(models)} repos...", file=sys.stderr)

    full = []
    for m in models:
        d = detail(m["modelId"])
        if d:
            d["downloads"] = m.get("downloads", 0)
            full.append(d)

    buckets = {"rocm": [], "nvfp4": [], "other": []}
    for m in full:
        buckets[classify(m["modelId"], m.get("tags") or [])].append(m)
    for v in buckets.values():
        v.sort(key=lambda x: x.get("downloads", 0), reverse=True)

    total_dl = sum(m.get("downloads", 0) for m in full)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    o = print
    o("# Published models")
    o("")
    o(f"**{len(full)} repositories** — {len(buckets['rocm'])} ROCmFP4/ROCmFPX for AMD Strix Halo, "
      f"{len(buckets['nvfp4'])} NVFP4 for NVIDIA, {len(buckets['other'])} other. "
      f"**{total_dl:,} downloads** in the last 30 days.")
    o("")
    o(f"Generated from the Hugging Face API on {stamp} by "
      "[`tools/gen_model_index.py`](tools/gen_model_index.py). Sizes are the sum of every "
      "file in the repository. Re-run the script to refresh.")
    o("")
    o("Full profile: [huggingface.co/kingjones777](https://huggingface.co/kingjones777)")
    o("")

    o("## ROCmFP4 / ROCmFPX — AMD Ryzen AI Max+ 395 (Strix Halo, gfx1151)")
    o("")
    o("> These files use the `Q4_0_ROCMFP4` / `Q*_0_ROCMFPX` tensor types, which are **not** "
      "stock llama.cpp quant types. They require a build of the "
      "[ROCmFPX fork](https://github.com/charlie12345/ROCmFPX) and will not load in upstream "
      "llama.cpp, Ollama, or LM Studio. See [recipes/quickstart.md](recipes/quickstart.md).")
    o("")
    o(HEADER)
    for m in buckets["rocm"]:
        o(row(m))
    o("")

    if buckets["nvfp4"]:
        o("## NVFP4 — NVIDIA (vLLM / SGLang)")
        o("")
        o(HEADER)
        for m in buckets["nvfp4"]:
            o(row(m))
        o("")

    if buckets["other"]:
        o("## Other formats")
        o("")
        o(HEADER)
        for m in buckets["other"]:
            o(row(m))
        o("")


if __name__ == "__main__":
    main()
