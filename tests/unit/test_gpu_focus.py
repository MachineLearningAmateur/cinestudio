"""GPU focus mode with a fake registry (no real registry changes)."""

import types

import pytest

from cinestudio import gpu_focus as g

FIREFOX = r"C:\Program Files\Mozilla Firefox\firefox.exe"
DISCORD = r"C:\Users\x\AppData\Local\Discord\app-1\Discord.exe"


class FakeKey:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_winreg(store, backup_path, writes):
    def enum(_k, i):
        items = list(store.items())
        if i >= len(items):
            raise OSError
        return items[i][0], items[i][1], 1

    def set_value(_k, name, _r, _t, value):
        writes.append(("set", name, backup_path.exists()))
        store[name] = value

    def delete(_k, name):
        writes.append(("delete", name, backup_path.exists()))
        del store[name]

    return types.SimpleNamespace(HKEY_CURRENT_USER=0, REG_SZ=1, OpenKey=lambda *a: FakeKey(),
                                 CreateKey=lambda *a: FakeKey(), EnumValue=enum, SetValueEx=set_value,
                                 DeleteValue=delete)


@pytest.fixture
def env(tmp_path, monkeypatch):
    store, writes = {}, []
    monkeypatch.setattr(g, "STATE_DIR", tmp_path)
    monkeypatch.setattr(g, "BACKUP", tmp_path / "backup.json")
    monkeypatch.setattr(g, "_reg", lambda: fake_winreg(store, tmp_path / "backup.json", writes))
    monkeypatch.setattr(g, "adapters", lambda: ["dGPU", "iGPU"])
    monkeypatch.setattr(g, "running_apps", lambda: {FIREFOX: "firefox", DISCORD: "Discord"})
    return store, writes


def test_backup_is_written_before_registry(env):
    store, writes = env
    g.on()
    assert writes and all(backup_existed for _, _, backup_existed in writes)
    assert store[FIREFOX] == g.POWER_SAVING


def test_off_restores_and_respects_user_edits(env):
    store, _ = env
    store[DISCORD.upper()] = "GpuPreference=2;"          # user had chosen the dGPU (other case)
    g.on()
    assert store[DISCORD.upper()] == g.POWER_SAVING        # existing spelling reused, no duplicate
    assert len(store) == 2
    store[FIREFOX] = "GpuPreference=0;"                    # user changes Firefox after `on`
    r = g.off()
    assert store[FIREFOX] == "GpuPreference=0;"            # left alone
    assert store[DISCORD.upper()] == "GpuPreference=2;"    # restored
    assert FIREFOX in r["left_unchanged_user_edits"]
    assert not g.BACKUP.exists()


def test_off_deletes_only_entries_it_created(env):
    store, _ = env
    g.on()
    g.off()
    assert store == {}


def test_off_leaves_entries_the_user_deleted(env):
    store, writes = env
    g.on()
    del store[FIREFOX]
    writes.clear()
    g.off()
    assert FIREFOX not in store
    assert all(name != FIREFOX for _, name, _ in writes)


def test_set_app_keeps_other_settings_and_reset_restores(env):
    store, _ = env
    exe = r"D:\Games\game.exe"
    store[exe] = "AutoHDREnable=1;GpuPreference=2;"
    g.set_app(exe, "integrated")
    assert g._parse(store[exe]) == {"AutoHDREnable": "1", "GpuPreference": "1"}     # HDR setting kept
    g.set_app(exe, "default")
    assert store[exe] == "AutoHDREnable=1;"
    g.set_app(exe, "gpu")
    g.off()
    assert store[exe] == "AutoHDREnable=1;GpuPreference=2;"                        # exactly as before


def test_set_app_refuses_our_tools_and_bad_choices(env):
    with pytest.raises(ValueError):
        g.set_app(r"C:\Python\python.exe", "integrated")
    with pytest.raises(ValueError):
        g.set_app(FIREFOX, "fast")


def test_choice_of_and_roles():
    assert g.choice_of("GpuPreference=1;") == "integrated"
    assert g.choice_of(None) == "default" and g.choice_of("AppStatus=0;") == "default"
    assert g.adapter_roles(["AMD Radeon(TM) Graphics", "NVIDIA GeForce RTX 3070"]) == \
        {"gpu": "NVIDIA GeForce RTX 3070", "integrated": "AMD Radeon(TM) Graphics"}
    assert g.adapter_roles(["NVIDIA GeForce RTX 3070"])["integrated"] is None


def test_a_failed_second_change_keeps_the_original_for_reset(env, monkeypatch):
    store, _ = env
    exe = r"D:\Apps\tool.exe"                                     # no entry at first
    g.set_app(exe, "integrated")
    real_write = g._write

    def broken(prefs, path, value):
        raise OSError("registry write failed")
    monkeypatch.setattr(g, "_write", broken)
    with pytest.raises(OSError):
        g.set_app(exe, "gpu")                                     # fails after the journal was updated
    monkeypatch.setattr(g, "_write", real_write)
    g.off()
    assert exe not in store                                       # the original (no entry) is back
