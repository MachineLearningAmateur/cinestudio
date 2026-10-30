"""Generation backend over the Comfy Developer Platform SDK (comfy-sdk).

Talks to a Comfy API v2 endpoint: comfy-api-proxy in front of a local or rented ComfyUI
(COMFY_BASE_URL, default http://127.0.0.1:8189). Input files are uploaded as SDK assets, so the
ComfyUI machine does not need to see local paths.

Submission is safe to retry: every submit carries a caller-chosen idempotency key, and the
graph carries a unique marker (run, clip, attempt) in its output file prefix. A reused key is
rejected by the server (IdempotencyKeyReuse); after an ambiguous failure the caller finds the job
with ``find(prefix)`` instead of submitting again. If the job cannot be found, ``submit`` raises
``UnresolvedSubmission``: the caller must record that state and must not start a new attempt
on its own (the job can still be in the proxy queue). (No job metadata labels: comfy-api-proxy 0.1.6
accepts metadata only as one string, comfy-sdk 0.5.0 sends a map.)
Never call ComfyUI's /free endpoint from here: unloading all models made ComfyUI reload ~37 GB
into 32 GB of RAM and crash (2026-10-18).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from comfy_sdk import Comfy
from comfy_sdk import exceptions as sdk_errors

DEFAULT_BASE_URL = "http://127.0.0.1:8189"
# comfy-sdk 0.5.0 final statuses (comfy_sdk._core.TERMINAL) + older spellings
TERMINAL = {"succeeded", "failed", "canceled", "expired", "cancelled", "completed"}
OOM_MARKERS = ("out of memory", "allocation on device", "outofmemory")


@dataclass
class JobState:
    id: str
    status: str
    error: str | None = None

    @property
    def done(self) -> bool:
        return self.status in TERMINAL

    @property
    def ok(self) -> bool:
        return self.status in ("succeeded", "completed")

    @property
    def oom(self) -> bool:
        return bool(self.error) and any(m in self.error.lower() for m in OOM_MARKERS)


class UnresolvedSubmission(RuntimeError):
    """The submit call failed in an ambiguous way and no job with the output prefix was found.
    The job may exist (queued in the proxy) or not. Record it; do not submit a new attempt
    until the user or a later ``find`` decides."""

    def __init__(self, prefix: str, key: str, cause: BaseException) -> None:
        super().__init__(f"submission of {prefix!r} (key {key!r}) is unresolved: {cause}")
        self.prefix, self.key, self.cause = prefix, key, cause


# Errors that prove the server did not accept the job: a new attempt is safe after these.
_REJECTED = tuple(getattr(sdk_errors, n) for n in ("InvalidWorkflow", "IdempotencyKeyReuse") if hasattr(sdk_errors, n))


class SdkBackend:
    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        # comfy-sdk reads the address only from COMFY_BASE_URL. An explicit base_url wins over the
        # environment; without one, an existing COMFY_BASE_URL wins over the local default.
        if base_url:
            os.environ["COMFY_BASE_URL"] = base_url
        else:
            os.environ.setdefault("COMFY_BASE_URL", DEFAULT_BASE_URL)
        self.base_url = os.environ["COMFY_BASE_URL"]
        self.client = Comfy(api_key=api_key) if api_key else Comfy()

    def submit(self, graph: dict[str, Any], files: dict[tuple[str, str], str | Path], key: str,
               prefix: str | None = None) -> str:
        """Upload `files` ({(node, input): path}) as assets, bind them, submit. Returns the job id.

        Raises sdk_errors.IdempotencyKeyReuse if `key` was used before, sdk_errors.InvalidWorkflow
        if the server rejects the graph. Any other error during the submit call is ambiguous: with
        `prefix` (the graph's unique output prefix) the backend looks for the job and returns its id,
        or raises UnresolvedSubmission."""
        wf = self.client.workflows.from_json(graph)
        for (node, field), path in files.items():
            wf.set_input(node, field, self.client.assets.from_file(str(path)))
        try:
            return self.client.submit(wf, idempotency_key=key).id
        except _REJECTED:
            raise
        except Exception as exc:  # noqa: BLE001 - transport errors: the job may or may not exist
            if prefix is None:
                raise
            try:
                found = self.find(prefix)
            except Exception:  # noqa: BLE001 - the server is not reachable: still unknown
                found = None
            if found:
                return found
            raise UnresolvedSubmission(prefix, key, exc) from exc

    @staticmethod
    def output_prefixes(graph: dict[str, Any]) -> set[str]:
        """The filename_prefix values of an API-format graph."""
        return {n["inputs"]["filename_prefix"] for n in graph.values()
                if isinstance(n, dict) and isinstance(n.get("inputs"), dict)
                and isinstance(n["inputs"].get("filename_prefix"), str)}

    def find(self, prefix: str, look_back: int = 50) -> str | None:
        """Newest recent job whose graph has an output node with filename_prefix == `prefix`
        (exact match, so run_c01_a1 does not match run_c01_a10), or None."""
        for summary in self.client.list_jobs(limit=look_back):
            try:
                wf = self.client.jobs.get(summary.id).get_workflow()
            except Exception:  # noqa: BLE001 - a job without a stored workflow cannot match
                continue
            graph = getattr(wf, "graph", None)
            if isinstance(graph, dict) and prefix in self.output_prefixes(graph):
                return summary.id
        return None

    def state(self, job_id: str) -> JobState:
        job = self.client.jobs.get(job_id)
        err = job.error
        return JobState(job.id, job.status, None if err is None else str(err))

    def watch(self, job_id: str, on_event: Callable[[str, dict[str, Any]], None]) -> JobState:
        """Stream events until the job ends; report progress through on_event(kind, data)."""
        job = self.client.jobs.get(job_id)
        for ev in job.events():
            name = type(ev).__name__
            if name == "Progress":
                on_event("progress", {"value": ev.value, "step": ev.step, "steps": ev.steps, "node": ev.current_node})
            elif name == "StatusChange":
                on_event("status", {"status": ev.status, "queue_position": ev.queue_position})
        # The SDK legitimately ends events() immediately when the server has no SSE support (501).
        # That does not mean the job ended: use the authoritative status endpoint until it does.
        while True:
            state = self.state(job_id)
            if state.done:
                return state
            time.sleep(2)

    def download(self, job_id: str, node_id: str, dst: str | Path) -> Path:
        outs = self.client.jobs.get(job_id).get_outputs(node_id)
        if not outs:
            # a mesh (SaveGLB: the "3d" output) is not an image or a video for the job client: the file is read from
            # ComfyUI's own history instead (MoGe panorama depth, 2026-10-22; used by the runs AND the character jobs)
            from . import server
            try:
                return server.download_output(server.comfyui_url(), job_id, node_id, Path(dst))
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"job {job_id}: no output on node {node_id} ({exc})") from exc
        return outs[0].to_file(str(dst))

    def cancel(self, job_id: str) -> JobState:
        """Job-scoped cancel (proxy -> ComfyUI /api/jobs/{id}/cancel): dequeue or interrupt this job only."""
        job = self.client.jobs.get(job_id).cancel()
        return JobState(job.id, job.status)


def is_key_reuse(exc: BaseException) -> bool:
    return isinstance(exc, getattr(sdk_errors, "IdempotencyKeyReuse", ()))


def marker_for(run: str, clip: int, attempt: int) -> str:
    """Unique, file-name-safe marker; put it in the output prefix of the clip's graph."""
    return f"{run}_c{clip:02d}_a{attempt}"


def iter_terminal(states: Iterable[JobState]) -> list[JobState]:
    return [s for s in states if s.done]
