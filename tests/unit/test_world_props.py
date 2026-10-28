"""bpy_jobs/plate_world.py and pano_world.py hide the template's set in a run but keep the character and the `Props`
collection (moving or touched objects, e.g. thrown balls). Loaded with a stand-in `bpy` (no Blender in the tests)."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

BPY_JOBS = Path(__file__).resolve().parents[2] / "bpy_jobs"


class Obj:
    def __init__(self, name, type_="MESH", parent=None):
        self.name, self.type, self.parent = name, type_, parent
        self.hide_render = self.hide_viewport = False


class Coll:
    def __init__(self, objs):
        self.all_objects = objs


def load(script, objects, collections, monkeypatch):
    bpy = types.ModuleType("bpy")
    bpy.data = types.SimpleNamespace(objects=objects, collections=collections)
    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = object
    mathutils.Matrix = mathutils.Quaternion = object
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", mathutils)
    spec = importlib.util.spec_from_file_location(f"_test_{script}", BPY_JOBS / f"{script}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scene():
    rig = Obj("MannequinRig", "ARMATURE")
    body = Obj("Sphere", parent=rig)
    proxy = Obj("Proxy_Mask")                            # a proxy kit part: in Mannequin, no rig parent
    balls = [Obj("Ball_Dodge1"), Obj("Ball_Kick1")]
    floor, wall, cart_ball = Obj("Plane"), Obj("Wall_Back"), Obj("Cart_Ball_01")
    cam, light, target = Obj("ShotCam", "CAMERA"), Obj("Key", "LIGHT"), Obj("CamTarget", "EMPTY")
    objects = [rig, body, proxy, *balls, floor, wall, cart_ball, cam, light, target]
    collections = {"Mannequin": Coll([rig, body, proxy]), "Props": Coll(balls), "Set": Coll([floor, wall, cart_ball])}
    return objects, collections


@pytest.mark.parametrize("script", ["plate_world", "pano_world"])
def test_hide_set_keeps_props_and_hides_the_set(script, monkeypatch):
    objects, collections = scene()
    mod = load(script, objects, collections, monkeypatch)
    hidden = mod.hide_set("Mannequin")
    assert sorted(hidden) == ["Cart_Ball_01", "Plane", "Wall_Back"]      # the static place: the background replaces it
    by = {o.name: o for o in objects}
    for n in ("Ball_Dodge1", "Ball_Kick1", "Sphere", "Proxy_Mask", "MannequinRig", "ShotCam", "Key", "CamTarget"):
        assert not by[n].hide_render and not by[n].hide_viewport, n
    for n in hidden:
        assert by[n].hide_render and by[n].hide_viewport


@pytest.mark.parametrize("script", ["plate_world", "pano_world"])
def test_hide_set_without_props_collection_is_unchanged(script, monkeypatch):
    objects, collections = scene()
    del collections["Props"]
    mod = load(script, objects, collections, monkeypatch)
    hidden = mod.hide_set("Mannequin")
    assert sorted(hidden) == ["Ball_Dodge1", "Ball_Kick1", "Cart_Ball_01", "Plane", "Wall_Back"]
