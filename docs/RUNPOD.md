# cinestudio on a rented RunPod GPU

A RunPod pod is already a container: start it from a stock image and run one install script. No image of our own is
needed (one can be built later from the same lock file, only to make new pods start faster). Not yet tested on a pod:
the first session checks the items in section 6.

## 1. Before you rent

- **Hugging Face:** the `HF_TOKEN` account must accept the licence of three gated repos (instant approval):
  `Lightricks/LTX-2.5`, `Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render`, `Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients`.
  Every other model file is public.
- **GitHub:** the repo is private (github.com/MachineLearningAmateur/cinestudio, branch `main`): make a fine-grained
  token with read-only "Contents" access to this one repo, for the clone on the pod.

## 2. Create the pod (RunPod console or the RunPod plugin)

- **Network volume** first, in the data center where the GPU is (the models live here and survive the pod). It mounts
  at `/workspace`. Model sizes from setup/models.json: 8gb tier 155.6 GB (164.5 GB with the optional `full_stage2`),
  large tier 233.3 GB. The runtime (ComfyUI venv with torch and the CUDA toolkit, Blender, the app venv) adds about
  11 GB, pip's cache (`PIP_CACHE_DIR=/workspace/.cache/pip` in RunPod images) 5-8 GB. Size: **8gb tier 200 GB, large
  tier 300 GB** (about 25-50 GB left for the cinestudio home and runs).
- **Pod:** a stock RunPod PyTorch image for CUDA 13.0 on Ubuntu 24.04, e.g.
  `runpod/pytorch:1.4.2-rc.184-cu1300-torch2121-ubuntu2404` (2026-10-22; its torch 2.12.1 is the pinned one). The
  default `--cuda cu130` needs a host driver >= 580: choose CUDA 13.0 in RunPod's CUDA filter (the script warns on an
  older driver). The image has Python 3.12/3.13 with venv, ffmpeg, git and tmux. The volume attached, SSH on.
- **GPU** for the large tier: 80 GB or more and about 120 GB of system RAM (the bf16 files: LTX + Gemma about 68 GB,
  Qwen 41 GB; on a 48 GB card they stream from RAM, 3-4x slower). Recommended: RTX PRO 6000 96 GB (best price per
  clip); H100 SXM 80 GB (about 6% faster, 60% more per hour). Not A100 (no fp8, slow int8), not 4090/5090 pods (35-41 GB
  RAM). docs/large_gpu_speed_research.md. The tier was called `24gb` before 2026-10-24; that name still works.
- Record the image's exact tag in `setup/runtime.lock.json` -> `pod.image` after the first successful test.

Renting costs money: nothing here creates resources by itself.

## 3. Put the code on the pod

```bash
git clone https://<github-token>@github.com/MachineLearningAmateur/cinestudio.git /workspace/cinestudio
```

Later code changes: push from the PC, then `git -C /workspace/cinestudio pull` on the pod. Without GitHub: copy a
tarball of the repo WITHOUT `.venv` and `.cinestudio` with `runpodctl send` / `runpodctl receive`. Never copy
`.cinestudio/` (private characters, runs, tokens).

## 4. Install (one command, about 10 min + the model download at ~100 MB/s or more)

Run it inside tmux: the model download takes most of an hour, and a closed SSH or web terminal stops it (a rerun
resumes the download).

```bash
tmux new -s install
cd /workspace/cinestudio
export HF_TOKEN=hf_...            # needed: the LTX-2.5 files (every tier) are gated; saved for later downloads too
bash setup/install_linux.sh --tier large      # or --tier 8gb; default --cuda cu130 (drivers >= 580); --cuda cu128 = torch 2.11, untested
```

It installs the pinned runtime (`setup/runtime.lock.json`): ComfyUI v0.42.1, only the 4 node packs the pipeline uses
(ComfyUI-LTXVideo, ComfyUI-GGUF, ComfyUI-segment-anything-2, comfyui_controlnet_aux), torch, SageAttention (optional),
Blender 5.2.0 (checksum checked); it links ComfyUI's models folder (and DWPose's checkpoints) to `/workspace/models`,
checks that file locks work in the cinestudio home, downloads the tier's models (SHA-256 checked, resumable), selects
the tier and runs `cinestudio doctor`. Doctor's ComfyUI and proxy checks fail at that point (the app starts them):
that is expected; read its other checks. Run the script again on a NEW pod with the same volume: it reinstalls the
runtime and reuses the models.

## 5. Use it

```bash
tmux new -s cinestudio            # the app keeps running when the SSH session ends (tmux attach -t cinestudio)
source /workspace/cinestudio-runtime/env.sh && cd /workspace/cinestudio && .venv/bin/python -m cinestudio app
```

From the PC: `ssh -L 8765:127.0.0.1:8765 <the pod's SSH address>`, then open http://127.0.0.1:8765. The LOCAL port must
be 8765 too: the app accepts only the host `127.0.0.1:8765` / `localhost:8765` (another local port gets 403 "unknown
host"). The app listens only on 127.0.0.1; do not expose port 8765 publicly.

Agents use the CLI on the pod as usual (`source .../env.sh` first). Without the app, start the proxy that the jobs go
through: `.venv/bin/python -m cinestudio proxy start`.

## 6. First pod session: check these (they cannot be checked on the PC)

- `free -g` vs the pod's RAM: the shared engine (large tier) now reads the container's cgroup limit, not the host's RAM.
- The install's flock check passed (the queue and job locks live on the network volume).
- Blender renders headless: Workbench and EEVEE need the NVIDIA EGL libraries (`NVIDIA_DRIVER_CAPABILITIES` must
  include `graphics`; RunPod's PyTorch images set `all`). Doctor runs only `blender --version`: make one short preview
  run.
- `cinestudio comfy start` turns on SageAttention for LTX (the engine's arguments list `--use-sage-attention`).
- The model download's free-disk check reports the volume's real free size.

## 7. Stop paying

Stop or terminate the pod when you are done; the volume keeps the models (it bills for its storage).
