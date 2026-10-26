"""The optional Civitai key and the Civitai download (cinestudio/civitai.py)."""
import pytest

from cinestudio import civitai, settings


def test_the_civitai_key_is_optional_and_never_shown(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CIVITAI_TOKEN", raising=False)
    st = settings.status()["civitai"]
    assert st["set"] is False and st["required"] is False and st["env"] == "CIVITAI_TOKEN"
    monkeypatch.setenv("CIVITAI_TOKEN", "abcdef0123456789secret")
    st = settings.status()["civitai"]
    assert st["set"] is True and st["source"] == "environment" and "secret" not in str(st)


def test_a_minimax_version_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(civitai, "version_info", lambda vid: {"baseModel": "MiniMax H3", "files": [{"name": "x.safetensors"}]})
    with pytest.raises(civitai.CivitaiError, match="MiniMax"):
        civitai.download(1, tmp_path)


def test_the_token_goes_to_civitai_but_not_to_the_cdn(tmp_path, monkeypatch):
    import hashlib
    import io
    import urllib.error
    import urllib.request
    data = b"lora-bytes"
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CIVITAI_TOKEN", "tok123")
    monkeypatch.setattr(civitai, "version_info", lambda vid: {"baseModel": "LTXV 2.5", "files": [
        {"name": "style.safetensors", "type": "Model", "downloadUrl": "https://civitai.com/api/download/models/1",
         "hashes": {"SHA256": hashlib.sha256(data).hexdigest().upper()}}]})
    seen = []

    class Opener:
        def open(self, req, timeout=None):
            seen.append((req.full_url, req.get_header("Authorization")))
            raise urllib.error.HTTPError(req.full_url, 307, "redirect", {"Location": "https://cdn.example/file"}, None)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: Opener())

    def urlopen(req, timeout=None, **kw):
        seen.append((req.full_url, req.get_header("Authorization")))
        return io.BytesIO(data)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    r = civitai.download(1, tmp_path)
    assert (tmp_path / "style.safetensors").read_bytes() == data and not r["already"]
    assert seen[0] == ("https://civitai.com/api/download/models/1", "Bearer tok123")
    assert seen[1] == ("https://cdn.example/file", None)              # the signed CDN address gets no token


def test_gpu_meters_come_back_after_a_driver_reset(monkeypatch):
    """After a failed NVML call the handle is dropped and NVML is started again (no app restart needed)."""
    import types
    from cinestudio import sysinfo
    calls = {"init": 0, "fail": True}

    def bad(*a):
        raise RuntimeError("GPU is lost")

    good = types.SimpleNamespace(gpu=50)
    mem = types.SimpleNamespace(used=1, total=8)
    nv = types.SimpleNamespace(
        nvmlInit=lambda: calls.__setitem__("init", calls["init"] + 1), nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda i: "h", NVML_TEMPERATURE_GPU=0,
        nvmlDeviceGetUtilizationRates=lambda d: bad() if calls["fail"] else good,
        nvmlDeviceGetMemoryInfo=lambda d: mem, nvmlDeviceGetTemperature=lambda d, t: 60, nvmlDeviceGetName=lambda d: "GPU")
    monkeypatch.setitem(__import__("sys").modules, "pynvml", nv)
    monkeypatch.setattr(sysinfo, "_nvml", None)
    monkeypatch.setattr(sysinfo, "_retry_at", 0.0)
    assert sysinfo.gpu() is None and sysinfo._nvml is None          # the reset: the handle is dropped
    calls["fail"] = False
    monkeypatch.setattr(sysinfo, "_retry_at", 0.0)                   # (the retry delay has passed)
    assert sysinfo.gpu()["util"] == 50.0 and calls["init"] == 2      # started again


def test_no_token_to_another_host_no_overwrite_and_a_hash_is_required(tmp_path, monkeypatch):
    """Codex review 2026-10-22: the token only for https://civitai.com; a different file of the same name is never
    replaced; a version without a SHA-256 is refused."""
    import hashlib
    import io
    import urllib.request
    data = b"lora-bytes"
    sha = hashlib.sha256(data).hexdigest()
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CIVITAI_TOKEN", "tok123")
    assert civitai._token_allowed("https://civitai.com/api/download/models/1")
    for bad in ("http://civitai.com/x", "https://evil.example/x", "https://civitai.com.evil.example/x",
                "https://civitai.com:8443/x"):
        assert not civitai._token_allowed(bad)
    info = {"baseModel": "LTXV 2.5", "files": [{"name": "style.safetensors", "type": "Model",
                                                "downloadUrl": "https://evil.example/f", "hashes": {"SHA256": sha}}]}
    monkeypatch.setattr(civitai, "version_info", lambda vid: info)
    seen = []

    class Opener:
        def open(self, req, timeout=None):
            seen.append(req.get_header("Authorization"))
            return io.BytesIO(data)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: Opener())
    civitai.download(1, tmp_path / "a")
    assert seen == [None]                                          # another host: no token
    own = tmp_path / "b" / "style.safetensors"
    own.parent.mkdir()
    own.write_bytes(b"the user's own file")
    with pytest.raises(civitai.CivitaiError, match="not replaced"):
        civitai.download(1, tmp_path / "b")
    assert own.read_bytes() == b"the user's own file"
    info["files"][0]["hashes"] = {}
    with pytest.raises(civitai.CivitaiError, match="SHA-256"):
        civitai.download(1, tmp_path / "c")


def test_a_broken_download_leaves_no_temp_file(tmp_path, monkeypatch):
    import hashlib
    import urllib.request
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CIVITAI_TOKEN", raising=False)
    monkeypatch.setattr(civitai, "version_info", lambda vid: {"baseModel": "LTXV 2.5", "files": [
        {"name": "style.safetensors", "type": "Model", "downloadUrl": "https://civitai.com/api/download/models/1",
         "hashes": {"SHA256": hashlib.sha256(b"x").hexdigest()}}]})

    class Broken:
        def read(self, n):
            raise TimeoutError("read timed out")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Opener:
        def open(self, req, timeout=None):
            return Broken()

    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: Opener())
    with pytest.raises(TimeoutError):
        civitai.download(1, tmp_path)
    assert list(tmp_path.glob("*.part")) == []
