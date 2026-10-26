"""CLI output contract: one JSON result line with "ok"; events are read only when complete."""

import json

from cinestudio import cli


def lines(capsys):
    return [json.loads(x) for x in capsys.readouterr().out.strip().splitlines()]


def test_usage_error_is_json(capsys):
    assert cli.main(["run", "nonsense"]) == 2
    (r,) = lines(capsys)
    assert r["ok"] is False and r["error_type"] == "UsageError"


def test_unknown_run_is_json(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path))
    assert cli.main(["run", "status", "r-missing"]) == 2
    (r,) = lines(capsys)
    assert r["ok"] is False and r["error_type"] == "RunError"


def test_unexpected_error_is_json_not_traceback(capsys, monkeypatch):
    monkeypatch.setattr(cli, "doctor", lambda: (_ for _ in ()).throw(OSError("disk on fire")))
    assert cli.main(["doctor"]) == 1
    (r,) = lines(capsys)
    assert r == {"ok": False, "error": "disk on fire", "error_type": "OSError"}


def test_half_written_event_is_not_read(tmp_path):
    f = tmp_path / "events.jsonl"
    f.write_bytes(b'{"event": "a"}\n{"event": "b"}\n{"event": "c", "par')
    assert [json.loads(x)["event"] for x in cli._complete_lines(f)] == ["a", "b"]
    f.write_bytes(f.read_bytes() + b'tial": 1}\n')
    assert len(cli._complete_lines(f)) == 3
