#!/usr/bin/env bash
# One-command install of cinestudio on a rented Linux GPU machine (a RunPod pod: the pod is already a container, no
# image of our own is needed). Idempotent: run it again and it only adds what is missing (a new pod on the same volume
# reuses the downloaded models).
#
#   bash setup/install_linux.sh --tier 8gb|large [--root /workspace/cinestudio-runtime] [--models /workspace/models]
#                               [--cuda cu130|cu128] [--no-sage] [--skip-models] [--yes]
#
# What it installs (pinned in setup/runtime.lock.json; ONLY what the current pipeline uses):
#   the app's Python venv (setup/requirements.txt), ComfyUI at its tested commit with its own venv and torch,
#   the 4 node packs the graphs need, optional SageAttention, Blender 5.2.0 (checksum checked), the models of the tier
#   (on the volume), then `cinestudio doctor`.
# Tokens: HF_TOKEN in the environment (a gated LTX-2.5 file needs it), or the script asks once (hidden input).
set -euo pipefail

TIER=""; ROOT="/workspace/cinestudio-runtime"; MODELS="/workspace/models"; CUDA="cu130"; SAGE=1; SKIP_MODELS=0; YES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tier) TIER="$2"; shift 2;;
    --root) ROOT="$2"; shift 2;;
    --models) MODELS="$2"; shift 2;;
    --cuda) CUDA="$2"; shift 2;;
    --no-sage) SAGE=0; shift;;
    --skip-models) SKIP_MODELS=1; shift;;
    --yes) YES=1; shift;;
    -h|--help) sed -n '2,16p' "$0"; exit 0;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
[ "$TIER" = "24gb" ] && { echo "note: the 24gb tier is now called 'large' (80 GB+ GPU)"; TIER="large"; }
[ -n "$TIER" ] || { echo "--tier 8gb or --tier large is needed" >&2; exit 2; }
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOCK="$REPO/setup/runtime.lock.json"
say() { printf '\n== %s\n' "$*"; }
lock() { python3 -c "import json,sys; d=json.load(open('$LOCK')); print(eval(sys.argv[1], {}, {'d': d}))" "$1"; }

say "checks"
command -v python3 >/dev/null || { echo "python3 is missing" >&2; exit 1; }
# the venvs need Python 3.12+ (numpy 2.5.4 in setup/requirements.txt requires it; Codex review 2026-10-24)
PY=""
for c in python3.13 python3.12 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "Python 3.12 or newer is needed (for example: apt install python3.12 python3.12-venv, or uv python install 3.12)" >&2; exit 1; }
echo "  Python: $($PY --version)"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  # cu130 torch needs a host driver >= 580 (CUDA 13.0); an older host gives torch.cuda.is_available() == False
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  if [ "$CUDA" = "cu130" ] && [ -n "$DRV" ] && [ "$DRV" -lt 580 ] 2>/dev/null; then
    echo "WARNING: the host driver is $DRV (< 580): cu130 torch will not see the GPU. Pick a pod with CUDA 13.0" \
         "(RunPod's CUDA filter, or a *-cu1300-* image) or use --cuda cu128 (untested)." >&2
  fi
else
  echo "warning: no nvidia-smi (no GPU?)"
fi
mkdir -p "$ROOT" "$MODELS"

if command -v apt-get >/dev/null && [ "$(id -u)" = "0" ]; then
  say "system packages (git, curl, Blender's libraries)"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq git curl xz-utils python3-venv libgl1 libegl1 libglib2.0-0 libxrender1 libxi6 \
    libxkbcommon0 libsm6 libxext6 libxxf86vm1 libxfixes3 libx11-6 >/dev/null
  # python3-venv covers only the distro's python3: the chosen python3.12/3.13 needs its own venv package
  if [ "$PY" != "python3" ]; then apt-get install -y -qq "${PY}-venv" >/dev/null 2>&1 || true; fi
fi

say "the app (venv + requirements)"
[ -x "$REPO/.venv/bin/python" ] || "$PY" -m venv "$REPO/.venv"
"$REPO/.venv/bin/python" -m pip install -q --upgrade pip
"$REPO/.venv/bin/python" -m pip install -q -r "$REPO/setup/requirements.txt"

say "ComfyUI $(lock "d['comfyui']['version']")"
CU="$ROOT/ComfyUI"
if [ ! -d "$CU/.git" ]; then git clone -q "$(lock "d['comfyui']['repo']")" "$CU"; fi
git -C "$CU" fetch -q origin
git -C "$CU" checkout -q "$(lock "d['comfyui']['commit']")"
[ -x "$CU/.venv/bin/python" ] || "$PY" -m venv "$CU/.venv"
CPY="$CU/.venv/bin/python"
"$CPY" -m pip install -q --upgrade pip
TORCH_VER=$(lock "d['torch']['linux_versions'].get('$CUDA', '')")
[ -n "$TORCH_VER" ] || { echo "--cuda must be one of: $(lock "' '.join(d['torch']['linux_versions'])")" >&2; exit 2; }
[ "$CUDA" = "cu130" ] || echo "note: --cuda $CUDA installs torch $TORCH_VER (untested; the tested build is $(lock "d['torch']['tested']"))"
if ! "$CPY" -c "import torch; assert torch.cuda.is_available() and torch.__version__.split('+')[0] == '$TORCH_VER'" 2>/dev/null; then
  # pinned: pip resolves torchvision / torchaudio to the releases made for this torch
  "$CPY" -m pip install -q "torch==$TORCH_VER" torchvision torchaudio --index-url "https://download.pytorch.org/whl/$CUDA"
fi
"$CPY" -m pip install -q -r "$CU/requirements.txt"

say "node packs (only the ones the pipeline uses)"
n=$(lock "len(d['node_packs'])")
for i in $(seq 0 $((n - 1))); do
  name=$(lock "d['node_packs'][$i]['name']"); repo=$(lock "d['node_packs'][$i]['repo']"); commit=$(lock "d['node_packs'][$i]['commit']")
  dst="$CU/custom_nodes/$name"
  [ -d "$dst/.git" ] || git clone -q "$repo" "$dst"
  git -C "$dst" fetch -q origin
  git -C "$dst" checkout -q "$commit"
  [ -f "$dst/requirements.txt" ] && "$CPY" -m pip install -q -r "$dst/requirements.txt"
  echo "  $name @ ${commit:0:7}"
done
if [ "$SAGE" = "1" ]; then "$CPY" -m pip install -q "sageattention==$(lock "d['sageattention']['tested']")" || echo "warning: SageAttention did not install (optional)"; fi

say "models folder on the volume ($MODELS)"
# every node pack that reads ComfyUI's own models folder (SAM 2 does) then sees the volume; nothing is copied
# safe_move SRC DST: copy every file of SRC that DST does not have, check each copy's size, and only then delete SRC
# (before: `cp -rn ... || true; rm -rf` could delete models after a failed copy: a full volume; Codex review 2026-10-24)
safe_move() {
  "$PY" - "$1" "$2" <<'PYCODE' || { echo "error: $1 was NOT deleted (the copy to $2 failed)" >&2; return 1; }
import os, shutil, sys
src, dst = sys.argv[1], sys.argv[2]
for root, _dirs, files in os.walk(src):
    rel = os.path.relpath(root, src)
    os.makedirs(os.path.join(dst, rel), exist_ok=True)
    for f in files:
        a, b = os.path.join(root, f), os.path.join(dst, rel, f)
        if os.path.exists(b):          # a file already on the volume wins (a rerun), as cp -n did; an interrupted
            continue                   # copy never has this name (it is written as .part and renamed after the check)
        part = b + ".part"
        shutil.copy2(a, part)
        if os.path.getsize(a) != os.path.getsize(part):
            sys.exit(f"copy check failed: {part} has another size than {a}")
        os.replace(part, b)
PYCODE
  rm -rf "$1"
}
if [ ! -L "$CU/models" ]; then
  safe_move "$CU/models" "$MODELS" || exit 1
  ln -s "$MODELS" "$CU/models"
fi
# DWPose's own checkpoints live in the pack folder: keep them on the volume too
AUX="$CU/custom_nodes/comfyui_controlnet_aux"
mkdir -p "$MODELS/controlnet_aux_ckpts"
if [ -d "$AUX" ] && [ ! -L "$AUX/ckpts" ]; then
  safe_move "$AUX/ckpts" "$MODELS/controlnet_aux_ckpts" || exit 1; ln -s "$MODELS/controlnet_aux_ckpts" "$AUX/ckpts"
fi

say "Blender $(lock "d['blender']['version']")"
BURL=$(lock "d['blender']['linux_url']"); BSHA=$(lock "d['blender']['sha256']")
BDIR="$ROOT/$(basename "$BURL" .tar.xz)"
if [ ! -x "$BDIR/blender" ]; then
  curl -fL --retry 3 -o "$ROOT/blender.tar.xz" "$BURL"
  echo "$BSHA  $ROOT/blender.tar.xz" | sha256sum -c -
  tar -xf "$ROOT/blender.tar.xz" -C "$ROOT" && rm -f "$ROOT/blender.tar.xz"
fi

say "environment ($ROOT/env.sh)"
cat > "$ROOT/env.sh" <<EOF
# source this before using cinestudio on this machine
export COMFYUI_DIR="$CU"
export CINESTUDIO_BLENDER="$BDIR/blender"
export CONTROLNET_AUX="$AUX"
export CINESTUDIO_HOME="\${CINESTUDIO_HOME:-/workspace/cinestudio-home}"
EOF
# shellcheck disable=SC1091
. "$ROOT/env.sh"
mkdir -p "$CINESTUDIO_HOME"
cd "$REPO"
APY="$REPO/.venv/bin/python"
"$APY" -m cinestudio comfy detect | tail -1

# the queue, job and engine locks live in CINESTUDIO_HOME (on the network volume): they need flock
if ! "$APY" - "$CINESTUDIO_HOME" <<'PY'
import fcntl, pathlib, sys
p = pathlib.Path(sys.argv[1]) / ".flock_test"
with open(p, "a+b") as fh:
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
p.unlink()
PY
then
  echo "WARNING: file locks (flock) do not work in $CINESTUDIO_HOME: set CINESTUDIO_HOME to a local disk in" \
       "$ROOT/env.sh (the queue and the jobs need them)." >&2
fi

if [ -n "${HF_TOKEN:-}" ]; then
  # keep it for later downloads from the app (Quick setup, optional features); the environment still wins
  printf '%s' "$HF_TOKEN" | "$APY" -m cinestudio settings token set hf | tail -1
elif [ "$YES" != "1" ] && [ -t 0 ]; then
  say "Hugging Face token (optional; the gated LTX-2.5 files need one with access) - press Enter to skip"
  read -rs -p "token: " tok; echo
  if [ -n "$tok" ]; then printf '%s' "$tok" | "$APY" -m cinestudio settings token set hf | tail -1; fi
  unset tok
fi

if [ "$SKIP_MODELS" != "1" ]; then
  say "models of the $TIER tier (resumable; checked by SHA-256; selects the tier when complete)"
  "$APY" -m cinestudio models download --tier "$TIER" | tail -1
fi

say "doctor"
# ComfyUI and the proxy are not running yet (the app starts them): their checks fail here, and that is expected
"$APY" -m cinestudio doctor | tail -1 || echo "note: doctor reports what is not running yet (ComfyUI, the proxy):" \
  "normal before the app starts; check its other lines (Blender, models, ffmpeg)"

cat <<EOF

Done. Start the app on the pod inside tmux (it keeps running when the SSH session ends):
  tmux new -s cinestudio
  source $ROOT/env.sh && cd $REPO && .venv/bin/python -m cinestudio app
From your PC, open a tunnel and the page (the local port must be 8765: the app accepts only that host):
  ssh -L 8765:127.0.0.1:8765 <the pod's SSH address>      then open http://127.0.0.1:8765
Agents that use only the CLI (no app): start the proxy first: .venv/bin/python -m cinestudio proxy start
EOF
