

def test_an_edit_without_a_reference_sends_picture_1_only():
    """Without a reference the edit graph sent the same picture again as Picture 2 (1.7x the time, bench 2026-10-24).
    With no reference upload, build removes Picture 2's nodes and the inputs that take it."""
    from cinestudio.workflow import ProfileWorkflow
    wf = ProfileWorkflow("qwenedit", "edit")
    values = {"instruction": "x", "seed": 1}
    img, ref = wf.targets("image")[0], wf.targets("reference")[0]
    one = wf.build(values, files={img: "a.png"})
    assert "ref" not in one and "ref_scale" not in one
    assert "image2" not in one["pos_txt"]["inputs"] and "image2" not in one["neg_txt"]["inputs"]
    two = wf.build(values, files={img: "a.png", ref: "b.png"})
    assert two["pos_txt"]["inputs"]["image2"] == ["ref_scale", 0] and "ref" in two
    assert "ref" in wf.build(values)                      # no files given: the graph as it is (old callers)


def test_the_large_tier_writes_larger_decode_tiles_into_a_runs_graphs(tmp_path):
    """The VAE decode tiles were 8 GB values in every tier. The large tier now writes larger tiles into a run's copy;
    the 8gb tier leaves the graphs as they are."""
    import json
    import shutil
    from cinestudio import models
    from cinestudio.workflow import WORKFLOWS, freeze_settings
    for name in ("ltx25_layout_api.json", "ltx25_layout.bind.json"):
        shutil.copy(WORKFLOWS / name, tmp_path / name)
    assert models.graph_settings("8gb") == {}
    used = freeze_settings(tmp_path, models.graph_settings("large"))
    assert used["VAEDecodeTiled"]["tile_size"] == 1024
    g = json.loads((tmp_path / "ltx25_layout_api.json").read_text(encoding="utf-8"))
    dec = [n for n in g.values() if n["class_type"] == "VAEDecodeTiled"]
    assert dec and all(n["inputs"]["tile_size"] == 1024 and n["inputs"]["temporal_size"] == 128 for n in dec)
