import argparse
import asyncio

import pytest

from cinestudio import cli, mcp_server

# Every CLI command (its leaf path) -> the MCP tool(s) that do the same. A new CLI command must be added here or to
# CLI_ONLY with the reason: the MCP server is the agent interface and must not fall behind the CLI.
CLI_TO_MCP = {
    "doctor": ["doctor"], "loras": ["list_loras"],
    "settings show": ["settings_status"],
    "models status": ["models_status"], "models download": ["download_models", "models_status"],
    "models stop": ["stop_model_download"],
    "comfy status": ["engine_status"],
    "queue list": ["queue_list"], "queue cancel": ["cancel_queued", "cancel_job", "cancel_run"],
    "background list": ["list_backgrounds", "show_background"], "background new": ["new_panorama"],
    "background plate": ["new_backplate"], "background from-set": ["backplate_from_set"],
    "background import": ["import_background"], "background depth": ["panorama_depth"],
    "background style": ["restyle_panorama"], "background delete": ["delete_background"],
    "character create": ["create_character"], "character import": ["import_character"],
    "character pick": ["pick_candidate"], "character edit": ["edit_image"], "character undo": ["undo_edit"],
    "character views": ["make_views"], "character redo-view": ["redo_view"], "character repose": ["repose_character"],
    "character style": ["restyle_sheet"], "character finish": ["finish_character"],
    "character proxy": ["set_proxy_kit", "preview_mannequin"], "character mannequin": ["set_mannequin_body"],
    "character list": ["list_characters"], "character show": ["show_character"],
    "character delete": ["delete_character"], "character rename": ["rename_character"],
    "character display-name": ["set_display_name"], "character use-style": ["set_character_style"],
    "character skin": ["character_skins"], "character skins": ["character_skins"],
    "character link-skin": ["character_skins"], "character unlink-skin": ["character_skins"],
    "job show": ["job_status"], "job list": ["list_jobs"],
    "prop list": ["list_props"], "prop show": ["show_prop"], "prop create": ["create_prop"],
    "prop import": ["import_prop"], "prop pick": ["pick_prop_candidate"], "prop edit": ["edit_prop"],
    "prop undo": ["undo_prop_edit"], "prop words": ["set_prop_words"], "prop use-style": ["set_prop_words"],
    "prop display-name": ["set_prop_words"], "prop skin": ["prop_skins"], "prop rename": ["rename_prop"],
    "prop delete": ["delete_prop"],
    "style list": ["list_styles"], "style show": ["show_style"], "style new": ["new_style"],
    "style edit": ["edit_style"], "style delete": ["delete_style"], "style set-default": ["set_default_style"],
    "style make-reference": ["make_style_reference"], "style use-reference": ["use_style_reference"],
    "workflow list": ["list_workflows", "show_workflow"], "workflow copy": ["copy_workflow"],
    "workflow set": ["set_workflow_value"], "workflow reset": ["reset_workflow"], "workflow use": ["use_workflow"],
    "film assemble": ["assemble_film"], "film list": ["list_films"],
    "song make": ["make_song"], "song list": ["list_songs"], "song show": ["show_song"], "song beats": ["song_beats"],
    "song loop": ["song_loop"], "song place": ["place_song"],
    "shot list": ["list_templates"], "shot show": ["show_template"], "shot add": ["add_template"],
    "shot layers": ["template_layers"], "shot timeline": ["template_timeline"], "shot clip-texts": ["template_clip_texts"],
    "shot check": ["check_template"],
    "shot from-video": ["template_from_video"], "shot pick-people": ["pick_template_people"],
    "shot make-dance": ["make_dance_template"], "shot delete": ["delete_template"],
    "run new": ["new_video_run"], "run start": ["continue_run"], "run fork": ["fork_run"],
    "run retake": ["retake_clip"], "run takes": ["list_takes"], "run use-take": ["use_take"],
    "run score": ["score_clip"], "run status": ["run_status", "run_details"], "run events": ["run_events"],
    "run list": ["list_runs"], "run cancel": ["cancel_run"], "run delete": ["delete_run"],
}
CLI_ONLY = {
    "app": "the user's own window: agents never start it (restart without --open only when asked)",
    "mcp": "this server itself",
    "setup": "interactive terminal setup that asks for tokens: tokens are never set through MCP",
    "settings token": "tokens are never set, checked with a value or shown through MCP (AGENTS.md section 2)",
    "settings tier": "the model tier for new work is the user's machine choice (app Settings or CLI)",
    "settings backend": "local vs Comfy Cloud (paid) is the user's choice: ask before anything that costs money",
    "settings models-dir": "machine configuration (the ComfyUI models folder), the user's choice",
    "models verify": "a blocking full SHA-256 pass over the tier (~150 GB, many minutes): run it in a terminal",
    "models work-download": "internal: the background download worker",
    "models civitai": "a third-party LoRA download: its licence must be checked by the user first (commercial-safe only)",
    "proxy": "starts/stops the comfy-api-proxy service: service control stays in the CLI",
    "comfy detect": "machine setup (finds and saves the ComfyUI start command)",
    "comfy start": "engine control is CLI-only: cinestudio starts the engine itself when a job needs it",
    "comfy stop": "engine control is CLI-only (a stop could end the user's other ComfyUI use)",
    "gpu list": "Windows graphics settings of the user's apps: only on the user's request (AGENTS.md section 2)",
    "gpu set": "changes the user's PC settings: only on request, by hand",
    "gpu all-integrated": "changes the user's PC settings: only on request, by hand",
    "gpu reset": "changes the user's PC settings: only on request, by hand",
    "sheet": "a low-level sheet helper; import_character(sheet=...) is the agent's way in",
    "job work": "internal: the background worker of a character job",
    "shot work": "internal: the background worker of a dance template",
    "shot work-mocap": "internal: the background worker of a template from a video",
    "shot work-mask": "internal: the background worker of the figure selection",
    "shot add-video": "video templates (the older video-to-video path); new work uses Blender templates",
    "shot mask": "the figure selection of a video template (the older video-to-video path)",
}


def _leaves(p: argparse.ArgumentParser, path=()):
    subs = [a for a in p._actions if isinstance(a, argparse._SubParsersAction)]
    if not subs:
        yield " ".join(path)
        return
    for name, sp in subs[0].choices.items():
        yield from _leaves(sp, path + (name,))


def _tools():
    return {t.name: t for t in asyncio.run(mcp_server.mcp.list_tools())}


def test_every_cli_command_has_a_tool_or_a_reason():
    leaves = set(_leaves(cli._parser()))

    def covered(leaf):
        return leaf in CLI_TO_MCP or any(leaf == k or leaf.startswith(k + " ") for k in CLI_ONLY)
    missing = sorted(x for x in leaves if not covered(x))
    assert not missing, f"CLI commands without an MCP tool (add one, or a reason in CLI_ONLY): {missing}"
    stale = sorted(set(CLI_TO_MCP) - leaves) + sorted(k for k in CLI_ONLY if not any(
        x == k or x.startswith(k + " ") for x in leaves))
    assert not stale, f"entries for CLI commands that no longer exist: {stale}"
    assert not set(CLI_TO_MCP) & set(CLI_ONLY)
    tools = _tools()
    for leaf, names in CLI_TO_MCP.items():
        for n in names:
            assert n in tools, f"{leaf}: no MCP tool {n}"
    assert all(reason.strip() for reason in CLI_ONLY.values())


def test_tools_and_the_agents_resource(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    tools = _tools()
    assert tools["cancel_run"].annotations.destructive_hint is True
    assert tools["queue_list"].annotations.read_only_hint is True
    assert "user asked" in tools["cancel_run"].description
    for name, t in tools.items():                       # every cancel, stop and delete is marked and says the rule
        if name.startswith(("cancel_", "delete_", "stop_")):
            assert t.annotations.destructive_hint is True, name
            assert "user asked" in " ".join(t.description.split()), name
        assert t.description and len(t.description) > 20, name
    assert "token" not in " ".join(tools)               # no tool sets or shows a token
    assert mcp_server.queue_list()["ok"] is True
    assert mcp_server.list_characters() == {"ok": True, "characters": []}
    assert "Hard rules" in mcp_server.agents_md()
    assert "Driving cinestudio through MCP" in mcp_server.agents_md()


def test_settings_status_never_shows_a_token(tmp_path, monkeypatch):
    from cinestudio import settings
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(settings, "status", lambda: {"hf": {"set": True, "source": "saved", "hint": "hf_…a4f2",
                                                            "label": "x"}})
    r = mcp_server.settings_status()
    assert r["tokens"] == {"hf": {"set": True, "source": "saved"}}


def test_a_failed_worker_start_is_not_reported_as_ok(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    blend = tmp_path / "shot.blend"
    blend.write_bytes(b"x")
    monkeypatch.setattr(mcp_server.character, "for_run", lambda name: ("sheet.png", "text"))
    monkeypatch.setattr(mcp_server.runner, "new_run", lambda *a, **k: {"id": "r1", "prompt": {"video": "v"},
                                                                       "character_name": "k"})
    monkeypatch.setattr(mcp_server.runner, "queue_label", lambda m: "label")
    monkeypatch.setattr(cli, "_start_detached", lambda rid: {"ok": False, "error": "worker failed"})
    r = mcp_server.new_video_run(str(blend), "k", "v")
    assert r["ok"] is False and r["id"] == "r1" and "worker failed" in r["error"]


def test_pick_and_finish_wait_in_the_queue(monkeypatch):
    calls = []
    monkeypatch.setattr(mcp_server.jobs, "submit", lambda action, name, **kw: calls.append((action, name, kw)) or {"id": "j"})
    mcp_server.pick_candidate("k", 7)
    mcp_server.finish_character("k")
    assert calls == [("pick", "k", {"via": "mcp", "seed": 7}), ("finish", "k", {"via": "mcp"})]


@pytest.fixture
def run_stubs(tmp_path, monkeypatch):
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    blend = tmp_path / "shot.blend"
    blend.write_bytes(b"x")
    seen = {"notes": [], "limits": []}

    def new_run(*a, **k):
        seen["args"], seen["kw"] = a, k
        return {"id": "r1"}
    monkeypatch.setattr(mcp_server.character, "for_run", lambda name: ("sheet.png", "text"))
    monkeypatch.setattr(mcp_server.runner, "new_run", new_run)
    monkeypatch.setattr(mcp_server.runner, "queue_label", lambda m: "label")
    monkeypatch.setattr(mcp_server.runner, "set_note", lambda rid, t: seen["notes"].append(t))
    monkeypatch.setattr(mcp_server.runner, "set_clip_limit", lambda rid, k: seen["limits"].append(k))
    monkeypatch.setattr(mcp_server.workqueue, "add", lambda *a: None)
    monkeypatch.setattr(cli, "_start_detached", lambda rid, *a: {"ok": True})
    seen["blend"] = str(blend)
    return seen


def test_new_video_run_passes_the_run_new_options(run_stubs):
    r = mcp_server.new_video_run(run_stubs["blend"], "k", "v", video_model="full", character2="b", spacing=1.2,
                                 look_scene="a gym", clip_videos={"2": "she bows"}, audio_relay="carry",
                                 sound_text="sneakers squeak", note="tests the duo", through_clip=1,
                                 advanced={"look_chain": "prev", "adain": 0.3, "character2_rig": "Rig2",
                                           "allow_external": True})
    assert r["ok"]
    kw = run_stubs["kw"]
    assert kw["video_model"] == "full" and kw["look_scene"] == "a gym" and kw["clip_videos"] == {"2": "she bows"}
    assert kw["character2"] == {"name": "b", "collection": None, "rig": "Rig2"} and kw["spacing"] == 1.2
    assert kw["look_chain"] == "prev" and kw["adain"] == 0.3 and kw["audio_relay"] == "carry"
    assert run_stubs["args"][8] is True                  # allow_external
    assert "background" not in kw                        # look_scene: no template default background
    assert run_stubs["notes"] == ["tests the duo"] and run_stubs["limits"] == [1]
    assert all(v is not None for k, v in kw.items() if k != "profile")   # None = the run default (left out)


def test_new_video_run_refuses_what_the_cli_refuses(run_stubs):
    with pytest.raises(ValueError, match="quality"):
        mcp_server.new_video_run(run_stubs["blend"], "k", "v", quality="ultra")
    with pytest.raises(ValueError, match="audio_relay"):
        mcp_server.new_video_run(run_stubs["blend"], "k", "v", audio_relay="loud")
    with pytest.raises(ValueError, match="unknown options"):
        mcp_server.new_video_run(run_stubs["blend"], "k", "v", advanced={"turbo": True})
    with pytest.raises(ValueError, match="look_chain"):
        mcp_server.new_video_run(run_stubs["blend"], "k", "v", advanced={"look_chain": "sideways"})
    with pytest.raises(ValueError, match="clip number"):
        mcp_server.new_video_run(run_stubs["blend"], "k", "v", clip_videos={"x": "t"})
    with pytest.raises(ValueError, match=r"characters\[0\]"):
        mcp_server.new_video_run(run_stubs["blend"], "k", "v", characters=["b", "k"])
    with pytest.raises(ValueError, match="not both"):
        mcp_server.new_video_run(run_stubs["blend"], "k", "v", characters=["k", "b"], character2="b")
    assert "kw" not in run_stubs                         # nothing was made


def test_new_video_run_passes_every_slot(run_stubs):
    assert mcp_server.new_video_run(run_stubs["blend"], "k", "v", characters=["k", "m", "k"])["ok"]
    assert run_stubs["kw"]["characters"] == [{"name": "k"}, {"name": "m"}, {"name": "k"}]   # twins are allowed
    assert "character2" not in run_stubs["kw"]
    assert mcp_server.new_video_run(run_stubs["blend"], "k", "v", characters=["k"])["ok"]
    assert "characters" not in run_stubs["kw"]


def test_new_video_run_takes_a_template_name(run_stubs, monkeypatch):
    from cinestudio import shots
    monkeypatch.setattr(shots, "load", lambda n: {"files": {"blend": run_stubs["blend"]}})
    monkeypatch.setattr(shots, "default_background", lambda b: None)
    assert mcp_server.new_video_run("shan_shen_bu_balls", "k", "v")["ok"]
    assert run_stubs["args"][0] == run_stubs["blend"]


def test_character_tools_queue_the_right_jobs(monkeypatch):
    calls = []
    monkeypatch.setattr(mcp_server.jobs, "submit", lambda action, name, **kw: calls.append((action, name, kw)) or {"id": "j"})
    monkeypatch.setattr(mcp_server.character, "load", lambda n: {"files": {"front": "f.png"}, "mannequin": {}})
    mcp_server.make_views("k", distance="wide", face="qwen")
    mcp_server.repose_character("k", seed=5)
    mcp_server.restyle_sheet("k", "lora.safetensors", keep=0.4)
    mcp_server.edit_image("k", "fix the hand", region="1,2,3,4", denoise=0.5, reference="crop.png")
    mcp_server.character_skins("k", new_style="anime_cel", seed=3)
    assert calls[0] == ("views", "k", {"via": "mcp", "mode": "quality", "seed": None, "distance": "wide shot",
                                       "face": "qwen"})
    assert calls[1] == ("repose", "k", {"via": "mcp", "shot": None, "seed": 5})
    assert calls[2] == ("style", "k", {"via": "mcp", "style_lora": "lora.safetensors", "keep": 0.4})
    assert calls[3][0] == "edit" and calls[3][2]["reference"] == "crop.png" and calls[3][2]["settings"] == {"denoise": 0.5}
    assert calls[4] == ("new_skin", "k", {"via": "mcp", "style": "anime_cel", "seed": 3})
    with pytest.raises(ValueError, match="distance"):
        mcp_server.make_views("k", distance="far")
    with pytest.raises(ValueError, match="target"):
        mcp_server.edit_image("k", "x", target="hat")
    monkeypatch.setattr(mcp_server.character, "load", lambda n: {"files": {"front": "f.png"}})
    with pytest.raises(ValueError, match="mannequin body"):
        mcp_server.repose_character("k")
    assert len(calls) == 5                               # refusals queue nothing


def test_import_and_instant_character_tools(monkeypatch):
    seen = []
    monkeypatch.setattr(mcp_server.character, "import_slots", lambda *a, **k: seen.append(("slots", a, k)) or {"name": "k"})
    monkeypatch.setattr(mcp_server.character, "import_sheet", lambda *a, **k: seen.append(("sheet", a, k)) or {"name": "k"})
    monkeypatch.setattr(mcp_server.character, "rescale_sheet", lambda n: seen.append(("rescale", n)) or {})
    monkeypatch.setattr(mcp_server.character, "set_display_name", lambda n, t: seen.append(("display", n, t)) or {})
    mcp_server.import_character("k", image="f.png", back="b.png", description="d")
    mcp_server.import_character("k", sheet="s.png")
    mcp_server.redraw_sheet("k")
    mcp_server.set_display_name("k", "Kit Sune")
    assert seen[0] == ("slots", ("k", {"front": "f.png", "back": "b.png"}, "d", "", "human"), {"display_name": ""})
    assert seen[1][0] == "sheet" and seen[2] == ("rescale", "k") and seen[3] == ("display", "k", "Kit Sune")
    with pytest.raises(ValueError, match="one of them"):
        mcp_server.import_character("k")
    with pytest.raises(ValueError, match="body"):
        mcp_server.import_character("k", image="f.png", body="robot")


def test_deletes_need_confirm(monkeypatch):
    called = []
    monkeypatch.setattr(mcp_server.character, "delete_character", lambda n: called.append(n) or {"deleted": n})
    monkeypatch.setattr(mcp_server.runner, "delete_run", lambda r: called.append(r) or {"deleted": r})
    for fn, arg in ((mcp_server.delete_character, "k"), (mcp_server.delete_run, "r1"),
                    (mcp_server.delete_background, "b"), (mcp_server.delete_template, "t"),
                    (mcp_server.delete_style, "s")):
        with pytest.raises(ValueError, match="confirm"):
            fn(arg)
    assert called == []
    assert mcp_server.delete_character("k", confirm=True)["deleted"] == "k"
    assert mcp_server.delete_run("r1", confirm=True)["deleted"] == "r1"


def test_background_tools_use_the_apps_queued_workers(tmp_path, monkeypatch):
    from cinestudio import shots
    from cinestudio.webapi import backgrounds as wb
    blend = tmp_path / "shot.blend"
    blend.write_bytes(b"x")
    seen = []
    monkeypatch.setattr(wb, "_new", lambda f: seen.append(("new", f)) or {"ref": "bg-new-x"})
    monkeypatch.setattr(wb, "_from_set", lambda f: seen.append(("set", f)) or {"ref": "bg-setplate-x"})
    monkeypatch.setattr(wb, "_style", lambda n, f: seen.append(("style", n, f)) or {"ref": "bg-style-x"})
    monkeypatch.setattr(wb, "_check_pano", lambda n: seen.append(("check", n)))
    monkeypatch.setattr(wb, "_start", lambda *a: seen.append(("start",) + a) or {"ref": "bg-depth-x"})
    monkeypatch.setattr(shots, "load", lambda n: {"files": {"blend": str(blend)}})
    assert mcp_server.new_panorama("a harbour", style_preset="3d_animated", depth=False)["ref"] == "bg-new-x"
    mcp_server.new_backplate("a gym", "dodge_tpl", seed=4)
    mcp_server.backplate_from_set("dodge_tpl", style="3d")
    mcp_server.panorama_depth("harbour")
    mcp_server.restyle_panorama("harbour", "l.safetensors", keep=0.2)
    assert seen[0] == ("new", {"scene": "a harbour", "style": "", "style_preset": "3d_animated", "seed": None,
                               "name": None, "depth": False})
    assert seen[1][1]["kind"] == "plate" and seen[1][1]["shot"] == str(blend) and seen[1][1]["seed"] == 4
    assert seen[2][0] == "set" and seen[2][1]["shot"] == str(blend)
    assert seen[3] == ("check", "harbour") and seen[4][:3] == ("start", "depth", "harbour")
    assert seen[5][:2] == ("style", "harbour") and seen[5][2]["keep"] == 0.2


def test_template_song_and_film_tools(monkeypatch):
    from cinestudio import film, shots, video_template
    from cinestudio.webapi import songs as ws
    seen = []
    monkeypatch.setattr(video_template, "start", lambda *a, **k: seen.append(("video", a, k)) or {"name": a[0]})
    monkeypatch.setattr(video_template, "pick", lambda n, t: seen.append(("pick", n, t)) or {"name": n})
    monkeypatch.setattr(shots, "start_dance", lambda *a: seen.append(("dance",) + a) or {"name": a[0]})
    monkeypatch.setattr(shots, "layers", lambda *a: seen.append(("layers",) + a) or {"layers": {}})
    monkeypatch.setattr(ws, "_song_new", lambda f: seen.append(("song", f)) or {"name": f["name"]})
    monkeypatch.setattr(film, "assemble", lambda *a, **k: seen.append(("film", a, k)) or {"film": "f.mp4"})
    mcp_server.template_from_video("duo", "d.mp4", seconds=8, people="pick", tracks="0,1")
    mcp_server.pick_template_people("duo", [0, 2])
    mcp_server.make_dance_template("d", "funk", loop=False, start=4.0)
    mcp_server.template_layers("t", set_description="a gym", props={"Ball_*": "red balls"})
    mcp_server.make_song("score", "orchestral, warm strings", seconds=40)
    mcp_server.assemble_film("f", ["r1", "r2"], score="score", duck=False)
    assert seen[0] == ("video", ("duo", "d.mp4", 0.0, 8, "video", ""), {"people": "pick", "tracks": "0,1"})
    assert seen[1] == ("pick", "duo", [0, 2])
    assert seen[2] == ("dance", "d", "funk", "S1,S2", False, 10.0, 4.0, False)
    assert seen[3] == ("layers", "t", "a gym", {"Ball_*": "red balls"}, False)
    assert seen[4][1]["seconds"] == 40 and seen[4][1]["lyrics"] == "[instrumental]"
    assert seen[5][1] == ("f", ["r1", "r2"]) and seen[5][2]["duck"] is False
    with pytest.raises(ValueError, match="people"):
        mcp_server.template_from_video("duo", "d.mp4", people="5")
    with pytest.raises(ValueError, match="sound"):
        mcp_server.template_from_video("duo", "d.mp4", sound="music")


def test_run_tools(monkeypatch):
    seen = []
    monkeypatch.setattr(mcp_server.runner, "fork_run", lambda *a, **k: seen.append(("fork", a, k)) or
                        {"id": "r2", "forked_from": "r1"})
    monkeypatch.setattr(mcp_server.runner, "add_take", lambda *a: seen.append(("take",) + a) or {"take": 1})
    monkeypatch.setattr(cli, "_start_detached", lambda rid, *a: seen.append(("start", rid)) or {"ok": True})
    r = mcp_server.fork_run("r1", 2, video="new text", audio_relay="off", note="why")
    assert r["ok"] and r["id"] == "r2"
    assert seen[0][1] == ("r1", 2, "why") and seen[0][2]["video"] == "new text" and seen[0][2]["audio_relay"] == "off"
    assert seen[1] == ("start", "r2")
    mcp_server.retake_clip("r1", 2, seed=9, advanced={"look_chain": "prev", "pose_stage2": "on"}, start=False)
    assert seen[2][0] == "take" and seen[2][3]["look_chain"] == "prev" and seen[2][3]["pose_stage2"] is True
    assert len(seen) == 3                                # start=False: no worker
    with pytest.raises(ValueError, match="keyframe"):
        mcp_server.fork_run("r1", 2, keyframe="maybe")
    with pytest.raises(ValueError, match="once"):
        mcp_server.retake_clip("r1", 2, seed=1, advanced={"seed": 2})


def test_run_events_reads_from_since(tmp_path, monkeypatch):
    d = tmp_path / "r1"
    d.mkdir()
    (d / "events.jsonl").write_text('{"event": "a"}\n{"event": "progress"}\n{"event": "b"}\n{"event": "partial',
                                    encoding="utf-8")
    monkeypatch.setattr(mcp_server.runner, "run_dir", lambda rid: d)
    r = mcp_server.run_events("r1")
    assert [e["event"] for e in r["events"]] == ["a", "b"] and r["next"] == 3
    assert mcp_server.run_events("r1", since=3)["events"] == []


def test_refusals_come_back_as_tool_errors_with_their_reason(tmp_path, monkeypatch):
    # the SDK turns a ToolError into an is_error result WITH its text, but hides the text of any other exception
    # ("Error executing tool X"): the server's _tool wrapper makes every refusal a ToolError, as readable as the CLI's
    from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
    monkeypatch.setenv("CINESTUDIO_HOME", str(tmp_path / "home"))
    with pytest.raises(ToolError, match="kind: 'everything' is not one of") as e:
        asyncio.run(mcp_server.mcp.call_tool("cancel_queued", {"kind": "everything", "ref": "x"}))
    assert not isinstance(e.value, UnexpectedToolError)
    with pytest.raises(ToolError, match="confirm=true"):
        asyncio.run(mcp_server.mcp.call_tool("delete_run", {"run_id": "r1"}))
    monkeypatch.setattr(mcp_server.runner, "status", lambda rid: (_ for _ in ()).throw(
        mcp_server.runner.RunError("no run r9")))
    with pytest.raises(ToolError, match="RunError: no run r9"):
        asyncio.run(mcp_server.mcp.call_tool("run_status", {"run_id": "r9"}))
    tools = _tools()                                     # the wrapper keeps the real parameters in the schema
    assert set(tools["cancel_queued"].input_schema["properties"]) == {"kind", "ref"}


def test_the_agents_md_guide_names_every_tool_and_the_count():
    import re
    text = mcp_server.agents_md()
    guide = text[text.index("### Driving cinestudio through MCP"):text.index("### Repository layout")]
    tools = set(_tools())
    assert tools <= set(re.findall(r"`([a-z_0-9]+)`", guide)), sorted(tools - set(re.findall(r"`([a-z_0-9]+)`", guide)))
    assert f"stdio; {len(tools)} tools" in text


def test_one_prompt_per_clip_through_the_mcp_and_the_cli(run_stubs, monkeypatch, capsys):
    import json
    r = mcp_server.new_video_run(run_stubs["blend"], "k", "a gym", clip_videos={"1": "it waves"},
                                 clip_context="faces the camera", character_words="the masked dancer")
    assert r["ok"] and r["prompt_warnings"] == []
    kw = run_stubs["kw"]
    assert kw["clip_context"] == "faces the camera" and kw["character_words"] == "the masked dancer"
    seen = []
    monkeypatch.setattr(mcp_server.runner, "fork_run", lambda *a, **k: seen.append(k) or {"id": "r2", "forked_from": "r1"})
    mcp_server.fork_run("r1", 2, clip_videos={"2": "it bows"})
    assert seen[0]["clip_videos"] == {"2": "it bows"}
    calls = []
    from cinestudio import shots
    monkeypatch.setattr(shots, "clip_texts", lambda name, texts=None, context=None: calls.append((name, texts, context))
                        or {"name": name, "clip_texts": texts or {}, "clip_context": context or "", "clips": []})
    assert mcp_server.template_clip_texts("gym", {"1": "x"}, "y")["ok"]
    assert cli.main(["shot", "clip-texts", "gym", "--clip", "2", "it kicks", "--context", ""]) == 0
    assert calls == [("gym", {"1": "x"}, "y"), ("gym", {"2": "it kicks"}, "")]
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["ok"] is True
    a = cli._parser().parse_args(["run", "new", "--blend", "b", "--character", "k", "--video", "v",
                                  "--clip-context", "", "--character-words", "the hero"])
    assert a.clip_context == "" and a.character_words == "the hero"
    f = cli._parser().parse_args(["run", "fork", "r1", "--from-clip", "2", "--clip-video", "2", "it bows"])
    assert f.clip_video == [["2", "it bows"]]
