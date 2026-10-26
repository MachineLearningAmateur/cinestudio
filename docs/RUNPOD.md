# cinestudio on a rented RunPod GPU

A RunPod pod is already a container: start it from a stock image and run one install script. No image of our own is
needed (one can be built later from the same lock file, only to make new pods start faster).

## 1. Create the pod (RunPod console or the RunPod plugin)

- **Network volume** first, in the data center where the GPU is (the models live here and survive the pod):
  8gb tier about 200 GB, large tier about 275 GB (the models alone are about 154 GB and 223 GB with every feature; the
  rest is for the cinestudio home and runs). It mounts at `/workspace`.
- **Pod:** a stock RunPod PyTorch image (Ubuntu, Python 3.12+; the script finds python3.13/3.12), the volume attached, SSH on. GPU for the large tier: 80 GB
  or more and about 120 GB of system RAM (the bf16 files: LTX + Gemma about 68 GB, Qwen 41 GB; on a 48 GB card they
  stream from RAM, 3-4x slower). Recommended: RTX PRO 6000 96 GB (best price per clip); H100 SXM 80 GB (about 6% faster,
  60% more per hour). Not A100 (no fp8, slow int8), not 4090/5090 pods (35-41 GB RAM). docs/large_gpu_speed_research.md.
  The tier was called `24gb` before 2026-10-24; that name still works.
- Record the image's exact tag in `setup/runtime.lock.json` -> `pod.image` after the first successful test.

Renting costs money: nothing here creates resources by itself.

## 2. Put the code on the pod

The repository has no remote yet. Either
- push it to a private GitHub repository and `git clone` it on the pod into `/workspace/cinestudio`, or
- copy it with the RunPod CLI: `runpodctl send cinestudio.tar.gz` on the PC (a tarball of the repo WITHOUT `.venv` and
  `.cinestudio`), `runpodctl receive <code>` on the pod, then unpack into `/workspace/cinestudio`.

Never copy `.cinestudio/` (private characters, runs, tokens).

## 3. Install (one command, about 10 min + the model download at ~100 MB/s or more)

```bash
cd /workspace/cinestudio
export HF_TOKEN=hf_...            # needed: the LTX-2.5 files (every tier) are gated; the token needs access
bash setup/install_linux.sh --tier large      # or --tier 8gb; default --cuda cu130 (drivers >= 580); --cuda cu128 = torch 2.11, untested
```

It installs the pinned runtime (`setup/runtime.lock.json`): ComfyUI v0.42.1, only the 4 node packs the pipeline uses
(ComfyUI-LTXVideo, ComfyUI-GGUF, ComfyUI-segment-anything-2, comfyui_controlnet_aux), torch, SageAttention (optional),
Blender 5.2.0 (checksum checked); it links ComfyUI's models folder (and DWPose's checkpoints) to `/workspace/models`,
downloads the tier's models (SHA-256 checked, resumable), selects the tier and runs `cinestudio doctor`.
Run it again on a NEW pod with the same volume: it reinstalls the runtime and reuses the models.

## 4. Use it

```bash
source /workspace/cinestudio-runtime/env.sh && cd /workspace/cinestudio && .venv/bin/python -m cinestudio app
```

From the PC: `ssh -L 8765:127.0.0.1:8765 <the pod's SSH address>`, then open http://127.0.0.1:8765 (the app listens
only on 127.0.0.1 and refuses other hosts; do not expose port 8765 publicly). Agents use the CLI on the pod as usual.

## 5. Stop paying

Stop or terminate the pod when you are done; the volume keeps the models (it bills for its storage).
