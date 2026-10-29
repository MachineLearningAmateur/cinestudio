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
case "$TIER" in 8gb|large) ;; *) echo "--tier 8gb or --tier large is needed" >&2; exit 2;; esac
case "$CUDA" in cu130) MIN_DRIVER=580;; cu128) MIN_DRIVER=570;; *) echo "--cuda must be cu130 or cu128" >&2; exit 2;; esac
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOCK="$REPO/setup/runtime.lock.json"
say() { printf '\n== %s\n' "$*"; }
lock() { "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {}, {"d": d}))' "$LOCK" "$1"; }

say "checks"
command -v python3 >/dev/null || { echo "python3 is missing" >&2; exit 1; }
# the venvs need Python 3.12+ (numpy 2.5.4 in setup/requirements.txt requires it; Codex review 2026-10-24)
PY=""
for c in python3.13 python3.12 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] in ((3, 12), (3, 13)) else 1)'; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "Python 3.12 or 3.13 is needed (the Linux dependency locks cover these versions)" >&2; exit 1; }
PY_VERSION=$("$PY" -c 'import sys; print("%s.%s" % sys.version_info[:2])')
"$PY" "$REPO/setup/check_linux.py" pins --python-version "$PY_VERSION" --cuda "$CUDA"
echo "  Python: $($PY --version)"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  # cu130 torch needs a host driver >= 580 (CUDA 13.0); an older host gives torch.cuda.is_available() == False
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  if [ -n "$DRV" ] && [ "$DRV" -lt "$MIN_DRIVER" ] 2>/dev/null; then
    echo "ERROR: the host driver is $DRV (< $MIN_DRIVER required for $CUDA). Pick a pod with a compatible" \
         "host driver, or select a supported --cuda build. Installation stopped before downloads." >&2
    exit 1
  fi
else
  echo "ERROR: nvidia-smi is missing; use an NVIDIA GPU pod with driver access." >&2
  exit 1
fi
export CINESTUDIO_HOME="${CINESTUDIO_HOME:-/workspace/cinestudio-home}"
mkdir -p "$ROOT" "$MODELS" "$CINESTUDIO_HOME"
# Fail before installs/downloads when the volume cannot enforce the queue's locks.
"$PY" "$REPO/setup/check_linux.py" locks --home "$CINESTUDIO_HOME"

check_venv() {
  actual=$("$1" -c 'import sys; print("%s.%s" % sys.version_info[:2])')
  [ "$actual" = "$PY_VERSION" ] || { echo "ERROR: $1 uses Python $actual; expected $PY_VERSION." \
    "Use a matching venv before rerunning (no existing venv was deleted)." >&2; exit 1; }
}

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
check_venv "$REPO/.venv/bin/python"
"$REPO/.venv/bin/python" -m pip install -q --require-hashes -r "$REPO/setup/linux/app-py$PY_VERSION.txt"
"$REPO/.venv/bin/python" -m pip check

say "ComfyUI $(lock "d['comfyui']['version']")"
CU="$ROOT/ComfyUI"
if [ ! -d "$CU/.git" ]; then git clone -q "$(lock "d['comfyui']['repo']")" "$CU"; fi
git -C "$CU" fetch -q origin
git -C "$CU" checkout -q "$(lock "d['comfyui']['commit']")"
[ -x "$CU/.venv/bin/python" ] || "$PY" -m venv "$CU/.venv"
CPY="$CU/.venv/bin/python"
check_venv "$CPY"
TORCH_VER=$(lock "d['torch']['linux_versions'].get('$CUDA', '')")
[ -n "$TORCH_VER" ] || { echo "--cuda must be one of: $(lock "' '.join(d['torch']['linux_versions'])")" >&2; exit 2; }
[ "$CUDA" = "cu130" ] || echo "note: --cuda $CUDA installs torch $TORCH_VER (untested; the tested build is $(lock "d['torch']['tested']"))"
# Resolve nothing at install time: the locks include the node packs and CUDA dependencies too.
"$CPY" -m pip install -q --require-hashes --extra-index-url "https://download.pytorch.org/whl/$CUDA" \
  -r "$REPO/setup/linux/comfy-py$PY_VERSION-$CUDA.txt"

say "node packs (only the ones the pipeline uses)"
n=$(lock "len(d['node_packs'])")
for i in $(seq 0 $((n - 1))); do
  name=$(lock "d['node_packs'][$i]['name']"); repo=$(lock "d['node_packs'][$i]['repo']"); commit=$(lock "d['node_packs'][$i]['commit']")
  dst="$CU/custom_nodes/$name"
  [ -d "$dst/.git" ] || git clone -q "$repo" "$dst"
  git -C "$dst" fetch -q origin
  git -C "$dst" checkout -q "$commit"
  echo "  $name @ ${commit:0:7}"
done
if [ "$SAGE" = "1" ]; then
  "$CPY" -m pip install -q --require-hashes --no-deps -r "$REPO/setup/linux/sageattention.txt" \
    || echo "warning: SageAttention did not install (optional)"
fi
"$CPY" -m pip check
say "CUDA in the completed ComfyUI environment"
"$CPY" "$REPO/setup/check_linux.py" cuda --torch-version "$TORCH_VER" --cuda "$CUDA"

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
# --skip-models defers downloads, but must still select the requested tier.
"$APY" -m cinestudio settings tier "$TIER" --force | tail -1

say "doctor"
# Only absent services are expected here; Blender, ffmpeg and completed model downloads must pass.
DOCTOR_ARGS=()
if [ "$SKIP_MODELS" = "1" ]; then DOCTOR_ARGS+=(--skip-models); fi
"$APY" "$REPO/setup/check_linux.py" doctor "${DOCTOR_ARGS[@]}"

cat <<EOF

Done. Start the app on the pod inside tmux (it keeps running when the SSH session ends):
  tmux new -s cinestudio
  source $ROOT/env.sh && cd $REPO && .venv/bin/python -m cinestudio app
From your PC, open a tunnel and the page (the local port must match the app's configured port):
  ssh -L 8765:127.0.0.1:8765 <the pod's SSH address>      then open http://127.0.0.1:8765
Agents that use only the CLI (no app): start the proxy first: .venv/bin/python -m cinestudio proxy start
EOF
