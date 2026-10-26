# cinestudio

Open-source, agent-first tool: a Blender shot (a placeholder mannequin and a moving camera) becomes
an AI video of **your own character**, from a reference image, in a style you choose. Generation
runs on open-weight models in ComfyUI through the Comfy Developer Platform SDK
(`comfy-sdk` -> `comfy-api-proxy` -> ComfyUI).

> Work in progress for the Comfy "Open Source Higgsfield" challenge. The JSON CLI, the web app and the
> MCP server work. [AGENTS.md](AGENTS.md) is the full manual (how to drive it, the rules, the tips).

## Layout

```
cinestudio/      core package: CLI, web app (web/, webapi/), MCP server, runs, characters, workflow bindings
bpy_jobs/        scripts that run inside Blender (renders, the mannequin body, the proxy kit)
blender_lib/     the mannequin and its looks (used by bpy_jobs)
templates/       built-in templates (library/) and the scripts that build shots from code
workflows/       one tested ComfyUI API graph per model profile + a binding map (.bind.json)
setup/           pinned requirements and runtime, model tiers with pinned revisions and hashes, Linux installer
tests/           unit tests (no GPU, no Blender) and integration scripts (need a running ComfyUI)
docs/            models, credits, RunPod, research reports
```

Local runtime files (characters, runs, templates, songs, backgrounds, tokens, proxy state) go to `.cinestudio/`
(git-ignored).

## Quick setup (tokens, model tier, download)

In the app: **Settings > Quick setup**. In a terminal (for example on a rented GPU machine):

```
python -m cinestudio setup                 # asks for the tokens, the tier, then downloads
python -m cinestudio settings token set hf # replace a token later (typed hidden, never on the command line)
python -m cinestudio models status         # what the tier needs and what is there
python -m cinestudio models download --detach
```

- **Hugging Face token**: faster downloads; needed for gated models. The LTX-2.5 files are gated, so a new install
  needs a token with access to Lightricks/LTX-2.5.
- **Comfy Cloud API key** (optional): only to run on Comfy Cloud (a paid Comfy plan). Local runs need no key.
- **Civitai API key** (optional): only for `models civitai` downloads that need it.
- **Model tier:** `8gb` (int8 models, about 156 GB of files with every feature, 8 GB VRAM, 32 GB RAM) or `large`
  (bf16 full models, about 225 GB of files, an 80 GB+ GPU and about 120 GB RAM; it was called `24gb` before, and that
  name still works). The default video (Layout-To-Render) uses the DISTILLED LTX-2.5 int8 model in both tiers; the
  full (dev) LTX-2.5 model is used for video-to-video (video templates). A run keeps the models it was made with.

Tokens are saved in `.cinestudio/secrets.json` (git-ignored, owner-only on Linux); `HF_TOKEN`, `COMFY_API_KEY` and
`CIVITAI_TOKEN` in the environment win over the saved values. The app and the CLI never show a token's value.

## Models

See [docs/MODELS.md](docs/MODELS.md). Download and check with:

```
python -m cinestudio models status --tier 8gb            # what the tier needs and what is there
python -m cinestudio models download --tier 8gb --detach # resumable, SHA-256 checked
python -m cinestudio models verify                       # full hash check of the files that are there
python -m cinestudio settings models-dir PATH            # optional: another models folder
```

## Using a shared ComfyUI

cinestudio loads one model family at a time (Z-Image, Qwen-Image-Edit, LTX-2.5 ...), because they do not
fit in 32 GB of RAM together. When the family changes, it restarts the ComfyUI engine that it started
itself (it stops the process and starts it again with the saved command). A ComfyUI that another program
started is never restarted automatically: cinestudio asks you to close it or restart it by hand
(`CINESTUDIO_RESTART=manager-shared` lets it restart such a ComfyUI through ComfyUI-Manager). It never
restarts ComfyUI while prompts that it did not submit are queued or running, and it never calls `/free`
or a global interrupt; a cancel stops only the run's own prompt.

This protection is best effort when people also use the same ComfyUI by hand: a prompt queued in
ComfyUI between cinestudio's queue check and its restart can still be stopped. Do not queue your
own prompts in the same ComfyUI while a cinestudio run is working, or set `CINESTUDIO_RESTART=none`
to restart ComfyUI yourself when the tool asks for it.

## The work queue

All GPU work (character steps, videos, songs, templates, backgrounds) waits in one first-in, first-out
queue, in the order of the requests. Every request runs in its own background process, so closing or restarting the app
does not stop it. Work stops early only by an explicit cancel:

```
python -m cinestudio queue list                      # what runs now and what waits
python -m cinestudio queue cancel character JOB_ID   # or: video RUN_ID, song NAME, template NAME, lab ID
```

The web app shows the same queue from the list button in the side rail, next to live CPU, RAM, GPU,
VRAM and GPU temperature meters.

## The ComfyUI engine

cinestudio needs a running ComfyUI, but not its window. It finds an installed ComfyUI one time
(Comfy Desktop, or a ComfyUI folder in `COMFYUI_DIR`), saves the start command in
`.cinestudio/comfyui.json`, and starts ComfyUI hidden in the background when the app opens or when a
job needs it. With Comfy Desktop it uses the same command, install, models and custom nodes.

```
python -m cinestudio comfy status     # running? started by cinestudio?
python -m cinestudio comfy start      # start it in the background
python -m cinestudio comfy stop       # only a ComfyUI that cinestudio started, with an empty queue
```

Set `CINESTUDIO_AUTOSTART_COMFYUI=0` to start ComfyUI yourself.

## Agents (MCP)

cinestudio is agent-first: every step is in the JSON CLI, and the same steps are served over MCP (stdio).
Add it to an MCP client, for example Claude Code:

```
claude mcp add cinestudio -- "<repo>/.venv/Scripts/python.exe" -m cinestudio mcp
```

or in a client's JSON config:

```json
{"mcpServers": {"cinestudio": {"command": "<repo>/.venv/Scripts/python.exe", "args": ["-m", "cinestudio", "mcp"],
                               "cwd": "<repo>"}}}
```

84 tools, one per CLI step: setup and models, the queue, styles, characters (create, import, edit, views,
repose, mannequin body, proxy kit, skins), backgrounds (backplates, panoramas), templates (from a video, dance, layers),
songs and films, videos (new run, clip gauntlet, fork) and workflow copies. The proxy kit (a character's clay stand-in
parts) is set only through MCP (`set_proxy_kit`); the app has no kit editor. Long steps go into the shared queue and
return a job or run id at once. The resource `cinestudio://agents` is AGENTS.md: the hard rules, the tips for good
output and the MCP guide (section 1, "Driving cinestudio through MCP").

## Credits

cinestudio is built on open-source software, open-weight models (LTX-2.5, Qwen-Image-Edit-2511, Z-Image,
Qwen-Image-2512, ACE-Step 1.5, SAM 3D Body and others) and published research: see [docs/CREDITS.md](docs/CREDITS.md)
(what each one is used for, its author and its licence, the licence obligations, and the research and techniques
behind the recipes). cinestudio ships no model file; the LTX-2.x Community License applies to the LTX-2.5 models
(free below USD 10 million annual revenue).
