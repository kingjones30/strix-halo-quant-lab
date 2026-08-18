#!/usr/bin/env bash
# Set up ROCmFP4 model serving on an AMD Ryzen AI Max+ 395 (Strix Halo, gfx1151).
#
# What this does:
#   1. Checks your hardware and ROCm install, and stops with a clear reason if
#      they are not what these models need.
#   2. Builds the ROCmFPX fork of llama.cpp (the only runtime that can read the
#      ROCmFP4 tensor types).
#   3. Downloads a model from https://huggingface.co/kingjones777
#   4. Starts an OpenAI-compatible server you can point LM Studio, Open WebUI,
#      Continue, Cline, or anything else at.
#
# What this does NOT do: make LM Studio load the .gguf file directly. It cannot.
# LM Studio ships stock llama.cpp, which rejects these tensor types outright
# ("invalid ggml type 100. should be in [0, 43)"). See the LM Studio section at
# the end for how to drive these models from the LM Studio interface anyway.
#
# Usage:
#   ./setup-strix-halo.sh                 # default model, build if needed
#   ./setup-strix-halo.sh --model ling    # pick a model
#   ./setup-strix-halo.sh --list          # show available models
#   ./setup-strix-halo.sh --check-only    # just run the hardware preflight
#   ./setup-strix-halo.sh --port 8080     # change the port (default 8080)
#
# MIT licensed. Model weights carry their own base-model licenses.

set -euo pipefail

# --progress-bar is great in a terminal and 245KB of noise in a log file.
if [[ -t 1 ]]; then CURL_PROGRESS=(--progress-bar); else CURL_PROGRESS=(-sS); fi

PREFIX="${STRIX_PREFIX:-$HOME/strix-halo}"
SRC_DIR="$PREFIX/ROCmFPX"
BIN_DIR="$SRC_DIR/build/bin"
MODEL_DIR="$PREFIX/models"
PORT=8080
HOST=127.0.0.1
MODEL_KEY=ling
CHECK_ONLY=0
BUILD_JOBS=4          # deliberate: higher values exhaust memory and fail the build

c_red=$'\033[31m'; c_grn=$'\033[32m'; c_ylw=$'\033[33m'; c_bld=$'\033[1m'; c_off=$'\033[0m'
ok()   { printf '%s  ok %s %s\n' "$c_grn" "$c_off" "$*"; }
warn() { printf '%s  !! %s %s\n' "$c_ylw" "$c_off" "$*"; }
die()  { printf '\n%s  STOP %s %s\n\n' "$c_red" "$c_off" "$*" >&2; exit 1; }
step() { printf '\n%s==> %s%s\n' "$c_bld" "$*" "$c_off"; }

# ---------------------------------------------------------------- model table
# repo : weights_file : size_GiB : draft_head_file ("-" if none) : description
# Every filename and size below was read from the Hugging Face API, not assumed.
declare -A MODELS=(
  [ling]="kingjones777/Ling-3.0-tiny-ROCmFP4-GGUF:Ling-3.0-tiny-Q4_0_ROCMFP4_COHERENT.gguf:4.30:-:Ling-3.0-tiny MoE - small and quick, good first test"
  [lfm2-8b]="kingjones777/LFM2-8B-A1B-ROCmFPX-GGUF:LFM2-8B-A1B-Q4_0_ROCMFP4_COHERENT.gguf:4.41:-:LFM2-8B-A1B - fastest model in the collection"
  [qwen]="kingjones777/Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF:Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf:13.75:mtp-Qwen3.8-27B-Q4_0.gguf:Qwen3.8-27B + MTP draft head - capable general model"
  [granite]="kingjones777/Granite-4.1-30B-ROCmFP4-GGUF:granite-4.1-30b-Q4_0_ROCMFP4_COHERENT.gguf:15.23:-:Granite-4.1-30b - dense 30B, general purpose"
)

list_models() {
  printf '\n%sAvailable models%s\n\n' "$c_bld" "$c_off"
  printf '  %-10s %-8s %s\n' KEY SIZE DESCRIPTION
  for k in ling lfm2-8b qwen granite; do
    IFS=: read -r _repo _file sz drafth desc <<< "${MODELS[$k]}"
    [[ "$drafth" == "-" ]] && extra="" || extra=" (+ draft head)"
    printf '  %-10s %-9s %s%s\n' "$k" "${sz} GiB" "$desc" "$extra"
  done
  printf '\n  Full index: https://github.com/kingjones30/strix-halo-quant-lab/blob/main/MODELS.md\n\n'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_KEY="${2:-}"; shift 2 ;;
    --port)  PORT="${2:-}"; shift 2 ;;
    --host)  HOST="${2:-}"; shift 2 ;;
    --list)  list_models; exit 0 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) die "Unknown option: $1  (try --help)" ;;
  esac
done

[[ -n "${MODELS[$MODEL_KEY]:-}" ]] || { list_models; die "Unknown model key: $MODEL_KEY"; }

# ------------------------------------------------------------------ preflight
step "Preflight: checking this machine can run ROCmFP4"

# ROCm installs to /opt/rocm/bin and system tools to /usr/sbin; neither is on the
# PATH of a non-login shell, so a bare `command -v rocminfo` gives a false negative.
for d in /opt/rocm/bin /opt/rocm-*/bin /usr/sbin /sbin; do
  [[ -d "$d" ]] && [[ ":$PATH:" != *":$d:"* ]] && PATH="$PATH:$d"
done
export PATH

[[ "$(uname -s)" == "Linux" ]] || die \
"These are Linux ROCm builds; this is $(uname -s).
 ROCmFP4 requires AMD ROCm and does not run on macOS or Windows."

command -v rocminfo >/dev/null 2>&1 || die \
"ROCm is not installed (no 'rocminfo' on PATH).
 Install ROCm 7.x first: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"

GFX="$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | sort -u | tr '\n' ' ' | sed 's/ $//')"
[[ -n "$GFX" ]] || die \
"ROCm is installed but reports no GPU.
 rocminfo found no gfx target. Check your driver and that your user is in the
 'render' and 'video' groups:  sudo usermod -aG render,video \$USER  (then log out and back in)"

ok "ROCm GPU target(s): $GFX"

if [[ "$GFX" != *gfx1151* ]]; then
  warn "Everything published in this collection was built and measured for gfx1151"
  warn "(Ryzen AI Max+ 395 / Strix Halo). You have: $GFX"
  warn "The build below targets gfx1151. To try your own target, set:"
  warn "    STRIX_GPU_TARGET=$(echo "$GFX" | awk '{print $1}') $0"
  warn "Results are unmeasured on that hardware."
  read -r -p "  Continue anyway? [y/N] " reply
  [[ "${reply,,}" == "y" ]] || die "Stopped. Nothing was changed."
fi
GPU_TARGET="${STRIX_GPU_TARGET:-gfx1151}"

RAM_GB=$(awk '/MemTotal/{printf "%.0f", $2/1048576}' /proc/meminfo)
ok "System memory: ${RAM_GB} GB"
[[ "$RAM_GB" -ge 32 ]] || warn "Under 32 GB; only the smallest models will fit."

MISSING_TOOLS=()
for t in git cmake curl make; do
  command -v "$t" >/dev/null 2>&1 || MISSING_TOOLS+=("$t")
done
if [[ ${#MISSING_TOOLS[@]} -gt 0 ]]; then
  # The AMD Ryzen AI Developer Platform image ships without cmake, so this is a
  # normal first-run stop rather than a broken machine.
  INSTALL_HINT="install them with your package manager"
  if command -v apt-get >/dev/null 2>&1; then
    INSTALL_HINT="sudo apt-get update && sudo apt-get install -y build-essential cmake git curl"
  elif command -v dnf >/dev/null 2>&1; then
    INSTALL_HINT="sudo dnf install -y gcc-c++ make cmake git curl"
  elif command -v pacman >/dev/null 2>&1; then
    INSTALL_HINT="sudo pacman -S --needed base-devel cmake git curl"
  fi
  die "Missing build tool(s): ${MISSING_TOOLS[*]}

 Install them, then run this again:

     $INSTALL_HINT"
fi
ok "Build tools present (git, cmake, curl, make)"

IFS=: read -r REPO FILE SIZE DRAFT DESC <<< "${MODELS[$MODEL_KEY]}"
AVAIL_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
NEED_GB=$(( ${SIZE%.*} + 12 ))   # model + build tree headroom
ok "Disk free in \$HOME: ${AVAIL_GB} GB (need roughly ${NEED_GB} GB)"
[[ "$AVAIL_GB" -ge "$NEED_GB" ]] || die "Not enough free disk: need ~${NEED_GB} GB, have ${AVAIL_GB} GB."

if [[ "$CHECK_ONLY" == "1" ]]; then
  printf '\n%s  Preflight passed. This machine can run ROCmFP4 models.%s\n\n' "$c_grn" "$c_off"
  exit 0
fi

# ---------------------------------------------------------------------- build
step "Building the ROCmFPX runtime"

if [[ -x "$BIN_DIR/llama-server" ]]; then
  ok "Already built: $BIN_DIR/llama-server"
else
  echo "  This is a one-time build and takes roughly 30-60 minutes."
  mkdir -p "$PREFIX"
  [[ -d "$SRC_DIR/.git" ]] || git clone --depth 1 https://github.com/charlie12345/ROCmFPX "$SRC_DIR"
  cd "$SRC_DIR"
  HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake -B build \
    -DGGML_HIP=ON -DGPU_TARGETS="$GPU_TARGET" -DGGML_HIP_ROCWMMA_FATTN=ON \
    -DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=OFF -DLLAMA_BUILD_WEBUI=OFF
  # -j 4 is deliberate. Higher job counts exhaust memory and fail this build.
  cmake --build build -j "$BUILD_JOBS"
  [[ -x "$BIN_DIR/llama-server" ]] || die "Build finished but $BIN_DIR/llama-server is missing."
  ok "Built $BIN_DIR/llama-server"
fi

# These binaries link against libggml/libllama in their own directory, so they
# cannot even start without LD_LIBRARY_PATH. Without it this check tests nothing
# and reports a perfectly good build as broken.
QUANTIZE_HELP="$(LD_LIBRARY_PATH="$BIN_DIR:/opt/rocm/lib" "$BIN_DIR/llama-quantize" --help 2>&1 || true)"
if ! grep -q ROCMFP4 <<< "$QUANTIZE_HELP"; then
  if grep -q "error while loading shared libraries" <<< "$QUANTIZE_HELP"; then
    die "The runtime cannot load its own libraries:

$(grep "error while loading" <<< "$QUANTIZE_HELP" | head -2)

 The build directory may be incomplete. Try rebuilding:  rm -rf $SRC_DIR/build"
  fi
  die "This build does not know the ROCmFP4 quant types. Something went wrong in the build."
fi
ok "Runtime understands the ROCmFP4 tensor types ($(grep -c ROCMFP4 <<< "$QUANTIZE_HELP") variants)"

# ------------------------------------------------------------------- download
step "Downloading model: $DESC"

mkdir -p "$MODEL_DIR"
TARGET="$MODEL_DIR/$FILE"
if [[ -f "$TARGET" ]]; then
  ok "Already downloaded: $TARGET"
else
  echo "  ${SIZE} GiB from https://huggingface.co/$REPO"
  URL="https://huggingface.co/$REPO/resolve/main/$FILE"
  curl -fL "${CURL_PROGRESS[@]}" -o "$TARGET.part" "$URL" || {
      rm -f "$TARGET.part"
      die "Download failed. Check the URL is reachable: $URL"; }
  mv "$TARGET.part" "$TARGET"
  ok "Downloaded to $TARGET"
fi

DRAFT_PATH=""
if [[ "$DRAFT" != "-" ]]; then
  DRAFT_PATH="$MODEL_DIR/$DRAFT"
  if [[ -f "$DRAFT_PATH" ]]; then
    ok "Draft head already present: $DRAFT"
  else
    echo "  fetching the MTP draft head ($DRAFT)"
    echo "  this is not optional - without it this model runs at roughly half speed"
    curl -fL "${CURL_PROGRESS[@]}" -o "$DRAFT_PATH.part" \
      "https://huggingface.co/$REPO/resolve/main/$DRAFT" || {
        rm -f "$DRAFT_PATH.part"; die "Draft head download failed."; }
    mv "$DRAFT_PATH.part" "$DRAFT_PATH"
    ok "Draft head downloaded"
  fi
fi

# ---------------------------------------------------------------------- serve
step "Starting the server on $HOST:$PORT"

if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  die "Port $PORT is already in use. Pick another with --port."
fi

# Speculative-decoding flags, straight from the model card. -n-max 4 is not a
# default: llama.cpp defaults to 16, which measures 19.17 tok/s against 30.30 at 4.
# --spec-draft-ngl 99 keeps the draft head on the GPU; without it the gain vanishes.
SPEC_FLAGS=""
if [[ -n "$DRAFT_PATH" ]]; then
  SPEC_FLAGS="--spec-type draft-mtp --model-draft $DRAFT_PATH \\
    --spec-draft-ngl 99 --spec-draft-device ROCm0 \\
    --spec-draft-n-max 4 --spec-draft-n-min 0 --spec-draft-p-min 0.0"
fi

cat > "$PREFIX/serve.sh" <<EOF
#!/usr/bin/env bash
# Start the ROCmFP4 server. Regenerated by setup-strix-halo.sh
exec env \\
  LD_LIBRARY_PATH="$BIN_DIR:/opt/rocm/lib" \\
  HSA_OVERRIDE_GFX_VERSION=11.5.1 \\
  GGML_HIP_ENABLE_UNIFIED_MEMORY=1 \\
  "$BIN_DIR/llama-server" \\
    --model "$TARGET" \\
    --host $HOST --port $PORT \\
    --n-gpu-layers 999 --flash-attn on -dio -fit off \\
    --ctx-size 32768 --cache-type-k q8_0 --cache-type-v q8_0 \\
    --jinja --no-mmap --alias "$MODEL_KEY" \\
    $SPEC_FLAGS "\$@"
EOF
chmod +x "$PREFIX/serve.sh"
ok "Wrote launcher: $PREFIX/serve.sh"

"$PREFIX/serve.sh" > "$PREFIX/server.log" 2>&1 &
SRV_PID=$!
printf '  waiting for the model to load'
for _ in $(seq 1 120); do
  sleep 2; printf '.'
  if curl -fs -m 2 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo; ok "Server is up (pid $SRV_PID)"
    break
  fi
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo
    echo "----- last 20 lines of $PREFIX/server.log -----"
    tail -20 "$PREFIX/server.log"
    die "The server exited while loading. Log above; full log at $PREFIX/server.log"
  fi
done

curl -fs -m 5 "http://$HOST:$PORT/health" >/dev/null 2>&1 \
  || die "Server did not become healthy in time. See $PREFIX/server.log"

step "Checking it actually generates"
REPLY_TEXT="$(curl -s -m 120 "http://$HOST:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"In one sentence: what is a B-tree?"}],"max_tokens":80}' \
  | python3 -c 'import sys,json
try:
    m=json.load(sys.stdin)["choices"][0]["message"]
    print(((m.get("content") or "") or (m.get("reasoning_content") or "")).strip()[:300])
except Exception as e:
    print("PARSE_FAILED:", e)' 2>/dev/null)"

if [[ -z "$REPLY_TEXT" || "$REPLY_TEXT" == PARSE_FAILED* ]]; then
  warn "The server is up but the test generation did not return text."
  warn "Check $PREFIX/server.log — the model may still be usable."
else
  ok "Model replied:"
  printf '       %s\n' "$REPLY_TEXT"
fi

# --------------------------------------------------------------------- how-to
cat <<EOF

$c_bld===============================================================$c_off
$c_grn  Running.$c_off  OpenAI-compatible API at:

      ${c_bld}http://$HOST:$PORT/v1${c_off}

  Manage it with:
      start :  $PREFIX/serve.sh
      stop  :  kill $SRV_PID
      log   :  tail -f $PREFIX/server.log
      swap  :  $0 --model <key>     ($0 --list)

$c_bld--- Using this from LM Studio -------------------------------$c_off

  LM Studio CANNOT open these .gguf files. Its bundled llama.cpp rejects
  the tensor types outright:

      invalid ggml type 100. should be in [0, 43)

  That is not fixable from your side, and importing the file will not help.
  Instead, let LM Studio talk to the server you just started:

    1. In LM Studio, open the plugin browser and install
       "openai-compat-endpoint" (lmstudio/openai-compat-endpoint).
    2. Set its base URL to:   http://$HOST:$PORT/v1
    3. Leave the API key blank, or put any placeholder in it.
    4. Chat as usual - LM Studio is now the front end, and this server
       is doing the inference.

  Note: that plugin step is GUI-only; there is no CLI for it.

$c_bld--- Or use anything else ------------------------------------$c_off

  Any tool with a "custom OpenAI base URL" field works with no plugin:
  Open WebUI, LibreChat, Continue, Cline, Aider, Cursor, your own code.

      base URL : http://$HOST:$PORT/v1
      API key  : anything (not checked)

  More models : $0 --list
  Full index  : https://github.com/kingjones30/strix-halo-quant-lab
$c_bld===============================================================$c_off

EOF
