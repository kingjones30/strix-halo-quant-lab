# scripts/

## `setup-strix-halo.sh`

One-command setup for running ROCmFP4 models on an AMD Ryzen AI Max+ 395
(Strix Halo, gfx1151). Checks your hardware, builds the ROCmFPX runtime,
downloads a model, and starts an OpenAI-compatible server.

```bash
./setup-strix-halo.sh --check-only     # does my machine qualify?
./setup-strix-halo.sh --list           # what can I run?
./setup-strix-halo.sh --model qwen     # set it up and serve
```

### What it will not do

It cannot make LM Studio, Ollama, or Jan open these `.gguf` files. Those ship
stock llama.cpp, which rejects the tensor types outright:

```
gguf_init_from_file_ptr: tensor 'blk.0.attn_k.weight'
  has invalid ggml type 100. should be in [0, 43)
```

What it does instead is start a server those tools — or any OpenAI-compatible
client — can talk to. See [`../recipes/lemonade.md`](../recipes/lemonade.md) for
the one front end that runs these models natively.

### Design notes

Things this script handles because they were hit during testing on real hardware:

- **`/opt/rocm/bin` and `/usr/sbin` are not on a non-login shell's PATH.** A bare
  `command -v rocminfo` reports "ROCm is not installed" on a machine that plainly
  has it. The script extends PATH before probing.
- **The AMD Ryzen AI Developer Platform image ships without `cmake`.** That's a
  normal first-run stop, so the script prints the exact install command for your
  package manager rather than failing obscurely.
- **The runtime binaries cannot start without `LD_LIBRARY_PATH`** pointing at
  their own build directory — they fail with `libllama-common.so.0: cannot open
  shared object file`. Any check that runs them without it tests nothing and will
  happily report a working build as broken.
- **Models that ship a draft head must be served with it.** Qwen3.8-27B measures
  30.30 tok/s with `--spec-draft-n-max 4` and 19.17 at llama.cpp's default of 16,
  so the script downloads the draft head and passes the documented flags. Serving
  it without the head is a configuration nobody should run.
- **Every filename and size in the model table is read from the Hugging Face API**,
  not transcribed by hand. One guessed filename in an early draft was a 404.

### Environment

| Variable | Purpose |
|---|---|
| `STRIX_PREFIX` | Where to build and store models (default `~/strix-halo`) |
| `STRIX_GPU_TARGET` | Override the GPU target (default `gfx1151`) |
