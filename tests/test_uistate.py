"""Guard rails for the remembered overlay geometry.

Every clamp here exists to stop the overlay coming back unreachable. It is
borderless and has no titlebar, so a window restored off-screen, at zero
opacity, or smaller than its own grip cannot be recovered with the mouse.
"""

import json

import pytest

from assistant import uistate
from assistant.uistate import UiState

# A single 1920x1080 screen at the origin, unless a test says otherwise.
ONE_SCREEN = (0, 0, 1920, 1080)


def _state(**kw) -> UiState:
    base = {"w": 460, "h": 300, "x": 40, "y": 40, "font_size": 12, "alpha": 1.0}
    return UiState(**{**base, **kw})


def test_missing_file_gives_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(uistate, "_state_path", lambda: tmp_path / "nope.json")
    assert uistate.load() == uistate.DEFAULT


def test_corrupt_file_gives_defaults_and_does_not_raise(tmp_path, monkeypatch):
    # A parse error on startup must not stop the app launching. Losing the
    # window size is a far smaller failure than not starting at all.
    path = tmp_path / "ui.json"
    path.write_text("{not json at all", encoding="utf-8")
    monkeypatch.setattr(uistate, "_state_path", lambda: path)
    assert uistate.load() == uistate.DEFAULT


def test_file_missing_keys_gives_defaults(tmp_path, monkeypatch):
    # Valid JSON, wrong shape: an older build, or a hand edit.
    path = tmp_path / "ui.json"
    path.write_text('{"w": 800}', encoding="utf-8")
    monkeypatch.setattr(uistate, "_state_path", lambda: path)
    assert uistate.load() == uistate.DEFAULT


def test_file_with_wrong_types_gives_defaults(tmp_path, monkeypatch):
    path = tmp_path / "ui.json"
    path.write_text(
        json.dumps({"w": "wide", "h": 300, "x": 0, "y": 0,
                    "font_size": 12, "alpha": 1.0}),
        encoding="utf-8",
    )
    monkeypatch.setattr(uistate, "_state_path", lambda: path)
    assert uistate.load() == uistate.DEFAULT


def test_round_trip(tmp_path, monkeypatch):
    path = tmp_path / "sub" / "ui.json"      # parent does not exist yet
    monkeypatch.setattr(uistate, "_state_path", lambda: path)
    saved = _state(w=700, h=520, x=120, y=90, font_size=18, alpha=0.6)
    uistate.save(saved)
    assert uistate.load() == saved


def test_save_failure_does_not_raise(tmp_path, monkeypatch):
    # Disk full, permissions, a locked file. Losing the setting is acceptable;
    # taking the app down on the quit path is not.
    def boom(*a, **k):
        raise OSError("nope")

    monkeypatch.setattr(uistate, "_state_path", lambda: tmp_path / "ui.json")
    monkeypatch.setattr(uistate.Path, "replace", boom)
    uistate.save(_state())          # must not raise


def test_alpha_zero_on_disk_is_pulled_back_to_the_floor():
    # The back door into the failure the live clamp exists to prevent: a
    # hand-edited or older file could otherwise restore an invisible window.
    got = _state(alpha=0.0).clamped(ONE_SCREEN)
    assert got.alpha == uistate.ALPHA_MIN
    assert got.alpha > 0.0


def test_alpha_above_one_is_pulled_down():
    assert _state(alpha=4.0).clamped(ONE_SCREEN).alpha == uistate.ALPHA_MAX


def test_font_size_out_of_range_is_clamped_both_ways():
    assert _state(font_size=999).clamped(ONE_SCREEN).font_size == uistate.FONT_MAX
    assert _state(font_size=1).clamped(ONE_SCREEN).font_size == uistate.FONT_MIN


def test_window_smaller_than_the_minimum_grows_back():
    got = _state(w=10, h=10).clamped(ONE_SCREEN)
    assert got.w == uistate.MIN_W
    assert got.h == uistate.MIN_H


def test_window_larger_than_the_desktop_shrinks():
    got = _state(w=99999, h=99999).clamped(ONE_SCREEN)
    assert got.w <= 1920
    assert got.h <= 1080


def test_position_far_off_screen_is_pulled_back():
    # The monitor it was saved on is gone. Without this the window is drawn
    # somewhere no mouse can reach and there is no titlebar to alt-drag.
    got = _state(x=9000, y=9000).clamped(ONE_SCREEN)
    assert got.x < 1920
    assert got.y < 1080
    assert got.x + got.w > 0
    assert got.y >= 0


def test_position_above_the_top_edge_is_pulled_down():
    # The drag bar is the top strip of the window. Negative y puts the only
    # surface that moves the window off the top of the screen.
    assert _state(y=-400).clamped(ONE_SCREEN).y >= 0


def test_a_position_on_a_second_monitor_is_left_alone():
    # The clamp must not yank a legitimately placed window back to the primary
    # screen, which would make multi-monitor setups worse, not safer.
    two = (0, 0, 3840, 1080)
    got = _state(x=2400, y=200).clamped(two)
    assert (got.x, got.y) == (2400, 200)


def test_a_monitor_left_of_the_primary_keeps_negative_coordinates():
    # Windows gives screens left of the primary negative origins. Clamping x
    # at zero would refuse a perfectly valid position.
    left_of = (-1920, 0, 1920, 1080)
    got = _state(x=-1500, y=100).clamped(left_of)
    assert (got.x, got.y) == (-1500, 100)


def test_clamped_returns_a_new_object():
    original = _state(alpha=0.0)
    got = original.clamped(ONE_SCREEN)
    assert original.alpha == 0.0, "clamped mutated its input"
    assert got is not original


def test_defaults_survive_their_own_clamp():
    # If the shipped default ever fell outside the guard rails, every fresh
    # install would silently start from a corrected value instead.
    assert uistate.DEFAULT.clamped(ONE_SCREEN) == uistate.DEFAULT


@pytest.mark.parametrize("bad", [(0, 0, 0, 0), (0, 0, -5, -5)])
def test_a_nonsense_desktop_still_yields_a_usable_window(bad):
    # If the desktop rect probe ever returns junk, the clamp must not produce a
    # zero-sized or negative window.
    got = _state().clamped(bad)
    assert got.w >= uistate.MIN_W
    assert got.h >= uistate.MIN_H
