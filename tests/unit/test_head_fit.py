"""The proxy kit's fitted head (head.fit, 2026-10-26): the kit check accepts a well-formed fit and refuses broken ones."""
import pytest

from cinestudio import character


def _fit(n=5):
    v = [1.0 - 0.5 * i for i in range(n)]
    return {"v": v, "a": [0.8] * n, "yf": [-1.0] * n, "yb": [1.0] * n,
            "mask": {"v0": 0.4, "v1": -1.0, "hw": [0.5] * n, "back": [-0.2] * n},
            "eye": [[0.2, -0.1], [0.5, -0.12], [0.3, -0.3]]}


def test_a_fitted_head_is_accepted():
    character.validate_proxy({"mask": {"color": [0.8, 0.8, 0.8]}, "hair": {"color": [0.04, 0.02, 0.05]},
                              "head": {"neck": [0.01, 0.01, 0.01], "fit": _fit()}})


@pytest.mark.parametrize("change", [
    lambda f: f.update(v=list(reversed(f["v"]))),            # levels must go from the top down
    lambda f: f.update(a=f["a"][:-1]),                         # one value per level
    lambda f: f["mask"].update(hw=[0.5, "x", 0.5, 0.5, 0.5]),  # numbers only
    lambda f: f.update(eye=[[0.1, 0.2, 0.3]]),                 # [u, v] points
    lambda f: f.update(v=[1.0, 0.5]),                          # at least 4 levels
])
def test_a_broken_fit_is_refused(change):
    fit = _fit()
    change(fit)
    with pytest.raises(character.CharacterError):
        character.validate_proxy({"head": {"fit": fit}})


def test_unknown_head_keys_are_refused():
    with pytest.raises(character.CharacterError, match="unknown keys"):
        character.validate_proxy({"head": {"shape": "egg"}})


def test_a_head_asset_is_checked(tmp_path):
    from cinestudio import character
    f = tmp_path / "head.blend"
    f.write_bytes(b"BLENDER")
    assert character.validate_proxy({"head": {"asset": {"file": str(f), "scale": 1.1}}})
    for bad in ({"file": str(tmp_path / "missing.blend")}, {"file": str(f), "colour": 1}, {"file": "head.glb"},
                {"file": str(f), "scale": 9}):
        try:
            character.validate_proxy({"head": {"asset": bad}})
        except character.CharacterError:
            continue
        raise AssertionError(bad)
