#!/usr/bin/env python3
"""
A/B benchmark harness for llama.cpp GGUF quants.

Compares two or more quants of the same model under conditions that make the
numbers actually comparable:

  * exactly N generated tokens per run (ignore_eos), so decode tok/s is comparable
  * a fresh UUID at the START of every prompt, so the server's prefix cache
    cannot serve a warm prefix and inflate prefill
  * cached_tokens recorded on every run so contamination is visible, not assumed
  * one model resident at a time, cold-loaded
  * multiple runs per cell, median reported

Usage:
    python3 ab_bench.py --config example-config.json
    python3 ab_bench.py --config example-config.json --dry-run

Writes <outdir>/results.json and prints a markdown table.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

# A neutral filler paragraph used to build prompts of a target token length.
FILLER = (
    "In a quiet valley between two weathered mountain ranges, a small river winds through "
    "meadows of tall grass. Travelers who follow the path along the bank notice wildflowers, "
    "the call of distant birds, and the scent of pine after rain. Local stories say an old "
    "bridge was built by artisans who understood both stone and water, and that the stones "
    "still remember their hands. At dusk the valley glows amber; at dawn it is silver with mist. "
)


def log(msg):
    print(msg, flush=True)


class Server:
    """Cold-starts one llama-server at a time and tears it down afterwards."""

    def __init__(self, cfg):
        self.binary = cfg["binary"]
        self.port = cfg.get("port", 18080)
        self.env = {**os.environ, **cfg.get("env", {})}
        self.extra_args = cfg.get("server_args", [])
        self.ctx = cfg.get("ctx_size", 32768)
        self.outdir = cfg["outdir"]
        # Only ever kills servers this harness started. Set true ONLY on a
        # machine you have to yourself - it will take down any other
        # llama-server on the box, including a live serving stack.
        self.kill_stray = cfg.get("kill_stray_servers", False)
        self.proc = None

    def _kill(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=30)
            self.proc = None
        if not self.kill_stray:
            return
        subprocess.run(["pkill", "-x", "llama-server"], check=False)
        for _ in range(30):
            if subprocess.run(["pgrep", "-x", "llama-server"],
                              capture_output=True).returncode != 0:
                return
            time.sleep(1)
        subprocess.run(["pkill", "-9", "-x", "llama-server"], check=False)
        time.sleep(2)

    def start(self, name, model_path):
        self._kill()
        logpath = os.path.join(self.outdir, f"server-{name}.log")
        cmd = [
            self.binary,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--model", model_path,
            "--ctx-size", str(self.ctx),
            "--n-gpu-layers", "999",
            "--no-warmup",
            "--jinja",
            "--alias", name,
        ] + self.extra_args
        log(f"  starting {name} ...")
        with open(logpath, "w") as fh:
            self.proc = subprocess.Popen(cmd, env=self.env, stdout=fh,
                                         stderr=subprocess.STDOUT)
        return self._wait_ready()

    def _wait_ready(self, timeout=1800):
        t0 = time.time()
        url = f"http://127.0.0.1:{self.port}/health"
        while time.time() - t0 < timeout:
            try:
                with urllib.request.urlopen(url, timeout=3) as r:
                    if r.status == 200:
                        dt = time.time() - t0
                        log(f"  ready in {dt:.0f}s")
                        return dt
            except Exception:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"server exited with code {self.proc.returncode} during load "
                    f"(see {self.outdir}/server-*.log)"
                )
            time.sleep(2)
        raise TimeoutError(f"server not ready after {timeout}s")

    def stop(self):
        self._kill()
        self.proc = None

    def chat(self, prompt, max_tokens, temperature=0.3):
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            # llama-server honours these as extra fields: force exactly max_tokens
            # so decode rates are comparable across models.
            "ignore_eos": True,
            "cache_prompt": False,
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as r:
            payload = json.load(r)
        wall = time.time() - t0
        return payload, wall

    def tokenize_count(self, text):
        """Exact prompt token count from the server's own tokenizer."""
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/tokenize",
            data=json.dumps({"content": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return len(json.load(r).get("tokens", []))


def build_prompt(server, target_tokens):
    """Grow the filler until the tokenizer reports ~target_tokens. Nonce goes FIRST."""
    reps = max(1, target_tokens // 60)
    text = FILLER * reps
    for _ in range(24):
        n = server.tokenize_count(text)
        if abs(n - target_tokens) <= max(64, target_tokens * 0.02):
            break
        ratio = target_tokens / max(n, 1)
        reps = max(1, int(reps * ratio))
        text = FILLER * reps
    return text


def measure(server, prompt_body, gen_tokens, runs):
    """Return list of per-run dicts. Nonce first => prefix cache cannot hit."""
    out = []
    for i in range(runs):
        nonce = uuid.uuid4().hex
        prompt = (
            f"[session {nonce}] Read the passage below, then write a detailed "
            f"continuation of it.\n\n{prompt_body}"
        )
        payload, wall = server.chat(prompt, gen_tokens)
        usage = payload.get("usage", {}) or {}
        timings = payload.get("timings", {}) or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        completion = usage.get("completion_tokens", 0)

        prefill = timings.get("prompt_per_second")
        decode = timings.get("predicted_per_second")
        if decode is None and completion:
            # Fallback: wall-clock. Less precise (includes prefill) but honest.
            decode = completion / wall

        row = {
            "run": i + 1,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": completion,
            "cached_tokens": cached,
            "prefill_tok_s": prefill,
            "decode_tok_s": decode,
            "wall_s": round(wall, 2),
            "timings_source": "server" if timings else "wall_clock",
        }
        if cached:
            row["WARNING"] = f"prefix cache served {cached} tokens - prefill invalid"
        log(f"    run {i+1}: prompt={row['prompt_tokens']} gen={completion} "
            f"cached={cached} prefill={_fmt(prefill)} decode={_fmt(decode)}")
        out.append(row)
    return out


def _fmt(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) else "n/a"


def median_of(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return statistics.median(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate config and model paths, then exit")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    cfg.setdefault("outdir", "./bench-out")
    os.makedirs(cfg["outdir"], exist_ok=True)

    gen = cfg.get("gen_tokens", 256)
    runs = cfg.get("runs", 3)
    targets = cfg.get("prompt_targets", [8000, 32000])
    models = cfg["models"]

    missing = [f"{n}: {p}" for n, p in models.items() if not os.path.exists(p)]
    if missing:
        log("Missing model files:")
        for m in missing:
            log(f"  {m}")
        sys.exit(1)
    if not os.path.exists(cfg["binary"]):
        log(f"Missing binary: {cfg['binary']}")
        sys.exit(1)
    if args.dry_run:
        log(f"Config OK. {len(models)} models, targets={targets}, "
            f"gen={gen}, runs={runs}. Nothing was started.")
        return

    server = Server(cfg)
    results = {"config": {k: cfg[k] for k in ("binary", "ctx_size", "server_args")
                          if k in cfg},
               "gen_tokens": gen, "runs": runs, "models": {}}

    try:
        for name, path in models.items():
            log(f"\n=== {name} ===")
            size_gib = os.path.getsize(path) / 2**30
            load_s = server.start(name, path)
            entry = {"path": os.path.basename(path),
                     "file_size_gib": round(size_gib, 2),
                     "load_seconds": round(load_s, 1),
                     "cells": {}}
            for target in targets:
                log(f"  building ~{target}-token prompt ...")
                body = build_prompt(server, target)
                log(f"  measuring @ ~{target} tokens")
                rows = measure(server, body, gen, runs)
                entry["cells"][str(target)] = {
                    "runs": rows,
                    "median_prefill_tok_s": median_of(rows, "prefill_tok_s"),
                    "median_decode_tok_s": median_of(rows, "decode_tok_s"),
                    "any_cache_hit": any(r.get("cached_tokens") for r in rows),
                }
            results["models"][name] = entry
            server.stop()
    finally:
        server.stop()

    outpath = os.path.join(cfg["outdir"], "results.json")
    with open(outpath, "w") as fh:
        json.dump(results, fh, indent=2)
    log(f"\nwrote {outpath}\n")
    print_table(results, targets)


def print_table(results, targets):
    hdr = ["Model", "Size (GiB)", "Load (s)"]
    for t in targets:
        hdr += [f"Prefill @{t//1000}K", f"Decode @{t//1000}K"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for name, e in results["models"].items():
        row = [name, f"{e['file_size_gib']:.2f}", f"{e['load_seconds']:.0f}"]
        for t in targets:
            c = e["cells"][str(t)]
            flag = " ⚠cache" if c["any_cache_hit"] else ""
            row += [f"{_fmt(c['median_prefill_tok_s'])}{flag}",
                    _fmt(c["median_decode_tok_s"])]
        print("| " + " | ".join(row) + " |")
    print("\nMedians of {} runs, exactly {} generated tokens per run."
          .format(results["runs"], results["gen_tokens"]))


if __name__ == "__main__":
    main()
