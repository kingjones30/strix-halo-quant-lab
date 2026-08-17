# Serving ROCmFP4 models through Lemonade

> **Status: verified working, 2026-08-17.** Lemonade Server 10.5.1 on Debian 13,
> Ryzen AI Max+ 395, serving `Ling-3.0-tiny-Q4_0_ROCMFP4_COHERENT.gguf` (a type-100
> FP4 model) through a custom ROCmFPX `llama-server`. Measured **98.5 / 99.1 / 101.6
> tok/s** end-to-end across three warm runs.

## Why this matters

ROCmFP4 files will not load in Ollama, LM Studio, Jan, or koboldcpp, because they all
bundle upstream llama.cpp and `Q4_0_ROCMFP4` is not an upstream tensor type (see
[the README](../README.md#why-wont-these-load-in-ollama-or-lm-studio)).

[Lemonade](https://github.com/lemonade-sdk/lemonade) is AMD's own open-source local LLM
server, and it is the one front end with a supported way out: it can be pointed at
**your own backend binary** instead of the one it downloads. That gives you an
OpenAI-compatible API, model management and a web UI in front of the ROCmFPX fork.

**The wrapper costs essentially nothing.** Same model, same machine:

| Path | Decode |
|---|---:|
| `llama-server` standalone (control) | 105.2 tok/s |
| Under Lemonade, llama-server internal timing | 102.6 – 105.3 tok/s |
| Under Lemonade, end-to-end over HTTP | 98.5 – 101.6 tok/s |

## Working recipe

### 1. Install Lemonade

```bash
sudo add-apt-repository ppa:lemonade-team/stable
sudo apt update
sudo apt install lemonade-server
```

The daemon is `lemond`; `lemonade` is only an HTTP client to it. To run an isolated
instance without touching system service state, give it its own cache directory and port:

```bash
setsid nohup env \
    HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    GGML_HIP_ENABLE_UNIFIED_MEMORY=1 \
    LD_LIBRARY_PATH=/path/to/rocmfpx \
  lemond ~/lemonade-test/cache --port 13399 --host 127.0.0.1 \
  > ~/lemonade-test/lemond.log 2>&1 &
```

**Start `lemond` with the ROCm environment set.** The `llama-server` child inherits it,
and that is how `HSA_OVERRIDE_GFX_VERSION` and unified memory reach the backend — there
is no config key for them.

### 2. Point it at your ROCmFPX build

```bash
lemonade --port 13399 config set llamacpp.rocm_bin=/path/to/rocmfpx/llama-server
lemonade --port 13399 config set llamacpp.backend=rocm
lemonade --port 13399 config set llamacpp.rocm_args="-fit off -np 1 --no-mmap"
```

**`rocm_bin` must be the path to the `llama-server` binary itself, not the directory
containing it.** AMD's docs say "path to your own backend binaries", which reads like a
directory; passing a directory produces `Failed to execute: <dir>` and a bare
`llama-server failed to start`.

`rocm_args` is merged into the launch line, so this is where mandatory flags like
`-fit off` go.

### 3. Pull the model from Hugging Face

```bash
lemonade --port 13399 pull \
  kingjones777/Ling-3.0-tiny-ROCmFP4-GGUF:Ling-3.0-tiny-Q4_0_ROCMFP4_COHERENT.gguf
```

### 4. Serve

The registered id is the repo name and filename joined, and it needs the `user.` prefix:

```bash
curl -s http://127.0.0.1:13399/api/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "user.Ling-3.0-tiny-ROCmFP4-GGUF-Ling-3.0-tiny-Q4_0_ROCMFP4_COHERENT.gguf",
    "messages": [{"role":"user","content":"What is a B-tree?"}],
    "max_tokens": 250
  }'
```

Verify the right binary is in use — the log names it explicitly:

```
[Debug] (llamacpp Server) Found executable at: /path/to/rocmfpx/llama-server
[Debug] (ProcessManager) Starting process with filtered output: /path/to/rocmfpx/llama-server
  -m .../Ling-3.0-tiny-Q4_0_ROCMFP4_COHERENT.gguf --ctx-size 4096 --port 8001 --jinja
  --context-shift --keep 16 --reasoning-format auto --no-webui -ngl 99 -fit off -np 1 --no-mmap
```

Note Lemonade supplies `-ngl 99` and takes context from its own `ctx_size` config
(default 4096), so raise that if you want a longer window:

```bash
lemonade --port 13399 config set ctx_size=32768
```

## Two bugs worth knowing about

Both cost real time, and both produce errors that point at the wrong thing.

**1. `extra_models_dir` models cannot be loaded.** Lemonade scans the directory and
registers each GGUF correctly — `downloaded: true`, `checkpoint` set to the right
absolute path. On load it then ignores all of that:

```
[Info] (Server) Auto-loading model: Ling-3.0-tiny-Q4_0_ROCMFP4_COHERENT
[Info] (Server) Model not cached, downloading from Hugging Face...
[Error] Failed to fetch model info from Hugging Face API (status: 404)
```

**2. Absolute paths in `user_models.json` get mangled into the HF cache namespace.**
A checkpoint of `/home/user/models/M.gguf` is rewritten by replacing separators with
dashes and looked up under the hub cache:

```
/home/user/.cache/huggingface/hub/models----home--user--models--M.gguf
```

That file does not exist, so llama-server reads nothing and reports:

```
gguf_init_from_file_ptr: failed to read magic
llama_model_load: error loading model
```

**`failed to read magic` reads exactly like a corrupt or unsupported model file.** It
is not — it is an empty-path read. If you see it under Lemonade, check the path in the
launch line before you suspect the quantization.

⇒ **Use `lemonade pull <repo>:<file>` and let Lemonade own the download.** The
HF-backed path works correctly; the local-file paths do not.

## Getting these models into Lemonade's built-in list

Unlike LM Studio, Lemonade has a real submission process. Per their
[FAQ](https://github.com/lemonade-sdk/lemonade/blob/main/docs/guide/faq.md), you can open
a pull request adding the model to the built-in `server_models.json`, or open an issue
requesting support. Maintainers review for device compatibility.

## Notes

Lemonade's ROCm stable channel already ships custom llama.cpp builds from
`lemonade-sdk/llama.cpp` — AMD maintains its own patched llama.cpp for AMD hardware.
That makes it a more plausible home for FP4 tensor-type support than upstream ggml-org,
and it is the same organization behind the
[AMD AI Developer Program](https://www.amd.com/en/developer/ai-dev-program.html) and
[amd/playbooks](https://github.com/amd/playbooks).
