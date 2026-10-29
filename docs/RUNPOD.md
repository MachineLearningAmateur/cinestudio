# cinestudio on a rented RunPod GPU

A RunPod pod is already a container: start it from a stock image and run one install script. No image of our own is
needed (one can be built later from the same lock file, only to make new pods start faster). Initial installation,
CUDA, Blender, model verification and preview and final videos passed on an RTX PRO 6000 pod. The remaining checks in
section 6, including a Qwen edit and a pod recreation, still need testing.

## 1. Before you rent

- **Hugging Face:** the `HF_TOKEN` account must accept the licence of three gated repos (instant approval):
  `Lightricks/LTX-2.5`, `Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render`, `Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients`.
  Every other model file is public.
- **GitHub:** github.com/MachineLearningAmateur/cinestudio, branch `main`, is public; cloning needs no token.

## 2. Create the pod (RunPod console or the RunPod plugin)

- **Network volume** first, in the data center where the GPU is (the models live here and survive the pod). It mounts
  at `/workspace`. Model sizes from setup/models.json: 8gb tier 155.6 GB (164.5 GB with the optional `full_stage2`),
  large tier 233.3 GB. The runtime (ComfyUI venv with torch and the CUDA toolkit, Blender, the app venv) adds about
  19 GiB on the first RTX PRO 6000 pod, and pip's cache can add several GB. Size: **8gb tier 200 GB, large
  tier 300 GB** (about 25-50 GB left for the cinestudio home and runs).
- **Pod:** a stock RunPod PyTorch image for CUDA 13.0 on Ubuntu 24.04, e.g.
  `runpod/pytorch:1.4.2-rc.184-cu1300-torch2121-ubuntu2404` (2026-10-22; its torch 2.12.1 is the pinned one). The
  default `--cuda cu130` needs a host driver >= 580: choose CUDA 13.0 in RunPod's CUDA filter (the script stops on an
  older driver). The image has Python 3.12/3.13 with venv, ffmpeg, git and tmux. The volume attached, SSH on.
- **GPU** for the large tier: 80 GB or more and about 120 GB of system RAM (the bf16 files: LTX + Gemma about 68 GB,
  Qwen 41 GB; on a 48 GB card they stream from RAM, 3-4x slower). Recommended: RTX PRO 6000 96 GB (best price per
  clip); H100 SXM 80 GB (about 6% faster, 60% more per hour). Not A100 (no fp8, slow int8), not 4090/5090 pods (35-41 GB
  RAM). docs/large_gpu_speed_research.md. The tier was called `24gb` before 2026-10-24; that name still works.
- Record the image's exact tag in `setup/runtime.lock.json` -> `pod.image` after the first successful test.

Renting costs money: nothing here creates resources by itself.

### Preload the volume before renting a GPU

The network volume can be populated with RunPod's [S3-compatible API](https://docs.runpod.io/storage/s3-api)
while no pod is running. US-NE-1 supports S3 and RTX PRO 6000 hosts. Create the volume in a region that supports
both S3 and the intended GPU; check GPU availability before choosing it. Storage bills during preparation, but
GPU compute starts billing only after a GPU pod is deployed.

Download and verify the tier on a PC or CPU machine first (`cinestudio models download` needs no GPU). Use an
isolated `CINESTUDIO_HOME` and models directory so selecting `large` does not change the PC app's default tier.
Reuse matching local weights after checking their size and SHA-256; download missing files into the staging area.
The large tier is for the 80-96 GB GPU setup, including the RTX PRO 6000 96 GB.

Upload to the network volume's root with these paths:

- Normal files: `models/<dir>/<file>` from `setup/models.json`.
- Entries under `@aux/ckpts/`: `models/controlnet_aux_ckpts/<remaining path>/<file>`; the installer links DWPose here.
- Code: use the public clone below, or transfer a current source archive when local changes have not been pushed.

S3 uses the volume ID as its bucket, the region's endpoint (US-NE-1: `https://s3api-us-ne-1.runpod.io`), and a separate
S3 key from RunPod Credentials. The access key is the `user_...` ID and the secret starts with `rps_...`; keep both
out of command lines, logs and the repository. Files above 500 MB require multipart uploads (AWS CLI/Boto3 handle
these). Check the remote sizes after upload. Then deploy the GPU pod with this volume attached and run the installer:
it reuses files already on the volume. Run `cinestudio models verify --tier large` on the pod before generation for
an end-to-end SHA-256 check. Runtime installation and loading the models still take some GPU-pod uptime.

## 3. Put the code on the pod

```bash
git clone https://github.com/MachineLearningAmateur/cinestudio.git /workspace/cinestudio
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
checks that file locks exclude another process and release correctly in the cinestudio home, downloads the tier's
models (SHA-256 checked, resumable), selects the tier and checks installation health. Only the ComfyUI and proxy
service checks are skipped before the app starts; a failed Blender, ffmpeg or model check stops installation.
`--skip-models` explicitly defers the model checks and still selects the requested tier.

The app and ComfyUI dependencies come from `setup/linux/`: exact versions and SHA-256 hashes, including transitive
dependencies and the CUDA-specific torch/torchvision/torchaudio builds. Python 3.12 and 3.13 on x86-64 Ubuntu 24.04
are covered; other Python versions stop before installation. `pip check` checks both finished environments, then a
CUDA gate imports the torch audio/vision extensions, checks the requested torch and CUDA versions, and verifies GPU
access **before model downloads**. A successful install does not yet prove that GPU rendering or generation works.
Run the script again on a NEW pod with the same volume to reuse the models.

Maintainers: after changing runtime pins or app requirements, run `python setup/lock_linux.py` with `uv` on PATH,
review the generated locks and test on a pod. `setup/linux/manifest.json` detects stale inputs or altered locks.
The initial Linux locks were resolved from the pinned upstream requirements, using the tested Windows ComfyUI
versions as constraints where available. Resolution succeeded for both Python versions and CUDA builds; GPU
execution remains unverified. Optional SageAttention is separately hash-pinned and cannot update the other packages.

## 5. Use it

```bash
tmux new -s cinestudio            # the app keeps running when the SSH session ends (tmux attach -t cinestudio)
source /workspace/cinestudio-runtime/env.sh && cd /workspace/cinestudio && .venv/bin/python -m cinestudio app
```

From the PC: `ssh -L 8765:127.0.0.1:8765 <the pod's SSH address>`, then open http://127.0.0.1:8765. The LOCAL port must
match the app's configured port: the app accepts only `127.0.0.1:<port>` / `localhost:<port>`. If the PC already runs
cinestudio on 8765, start the pod app with `app --port 8766` and tunnel `8766:127.0.0.1:8766` instead. The app listens
only on 127.0.0.1; do not expose it publicly.

Agents use the CLI on the pod as usual (`source .../env.sh` first). Without the app, start the proxy that the jobs go
through: `.venv/bin/python -m cinestudio proxy start`.

## 6. First pod session: check these (they cannot be checked on the PC)

- `free -g` vs the pod's RAM: the shared engine (large tier) now reads the container's cgroup limit, not the host's RAM.
- The install's two-process lock check passed (the queue and job locks live on the network volume).
- Blender renders headless: Workbench and EEVEE need the NVIDIA EGL libraries (`NVIDIA_DRIVER_CAPABILITIES` must
  include `graphics`; RunPod's PyTorch images set `all`). On an idle pod, after sourcing `env.sh`, run
  `.venv/bin/python setup/check_linux.py render`. It renders a 64px factory-scene frame with each engine and retains
  the PNGs and log in `<home>/cache/setup_checks/`. It refuses while work waits or runs and holds the GPU lock.
  Inspect both PNGs, then make one short preview video from a built-in template and character.
- The built-in `preview.mp4` is for browsing. Runs need layout RGB, masks, depth and joint positions after applying
  the selected characters' bodies and proxy kits. These are rendered on the pod and cached for matching later runs;
  pre-rendered control bundles are not shipped in the template library.
- An LTX job turns on SageAttention when installed (the engine's arguments list `--use-sage-attention`). A bare
  `cinestudio comfy start` has no known model family and leaves SageAttention off.
- The model download's free-disk check reports the volume's real free size.
- Run a Qwen edit followed by an LTX preview; check both outputs, family switching and peak container RAM. On large
  pods with 150 GB+ RAM the shared engine keeps families loaded, so check memory use across successive jobs.
- Recreate a pod on the same network volume and rerun installation: models should be reused and the selected tier
  should remain correct. Record the working image tag and digest in `setup/runtime.lock.json` and regenerate locks.

## 7. Stop paying

Stop or terminate the pod when you are done; the volume keeps the models (it bills for its storage).
