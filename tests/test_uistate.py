"""Guard rails for the remembered overlay geometry.

Every clamp here exists to stop the overlay coming back unreachable. It is
borderless and has no titlebar, so a window restored off-screen, at zero
opacity, or smaller than its own grip cannot be recovered with the mouse.
"""

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from assistant import uistate
from assistant.uistate import UiState

# A single 1920x1080 screen at the origin, unless a test says otherwise.
ONE_SCREEN = [(0, 0, 1920, 1080)]


def _grabbable(got, monitors) -> bool:
    """Independent restatement of "you can still grab this window".

    Deliberately not uistate._reachable_on. Using the implementation as its own
    oracle means a mutation changes the assertion too, and the test passes on
    broken code - which is exactly what happened before this was written out
    by hand.
    """
    return any(
        top <= got.y <= bottom - 90
        and min(got.x + got.w, right) - max(got.x, left) >= 90
        for left, top, right, bottom in monitors
    )


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


@pytest.mark.parametrize("value", ["1e400", "Infinity", "-Infinity", "NaN"])
def test_numbers_python_json_accepts_but_int_cannot_hold(tmp_path, monkeypatch, value):
    # json.loads accepts Infinity and NaN, which are not JSON but are Python's
    # dialect of it, and 1e400 parses to inf. int(inf) raises OverflowError and
    # int(nan) raises ValueError - neither is a parse error, so a narrow except
    # around json.loads misses them and the app refuses to start over a
    # remembered window size.
    path = tmp_path / "ui.json"
    path.write_text(
        f'{{"w": {value}, "h": 300, "x": 0, "y": 0, "font_size": 12, "alpha": 1.0}}',
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


def test_position_off_only_the_right_edge_is_pulled_back():
    # x alone is off-screen, y is fine. The combined case cannot fail on a
    # broken horizontal check, because the vertical one rejects it first.
    got = _state(x=9000, y=100).clamped(ONE_SCREEN)
    assert _grabbable(got, ONE_SCREEN), f"x={got.x} leaves nothing to grab"
    assert got.x < 1920


def test_position_above_the_top_edge_is_pulled_down():
    # The drag bar is the top strip of the window. Negative y puts the only
    # surface that moves the window off the top of the screen.
    assert _state(y=-400).clamped(ONE_SCREEN).y >= 0


def test_a_position_on_a_second_monitor_is_left_alone():
    # The clamp must not yank a legitimately placed window back to the primary
    # screen, which would make multi-monitor setups worse, not safer.
    two = [(0, 0, 1920, 1080), (1920, 0, 3840, 1080)]
    got = _state(x=2400, y=200).clamped(two)
    assert (got.x, got.y) == (2400, 200)


def test_a_monitor_left_of_the_primary_keeps_negative_coordinates():
    # Windows gives screens left of the primary negative origins. Clamping x
    # at zero would refuse a perfectly valid position.
    left_of = [(0, 0, 1920, 1080), (-1920, 0, 0, 1080)]
    got = _state(x=-1500, y=100).clamped(left_of)
    assert (got.x, got.y) == (-1500, 100)


def test_a_position_in_the_dead_space_of_an_L_shaped_layout_is_moved():
    # Two monitors diagonal to each other. Their bounding box covers a region
    # belonging to neither, so a union check passes a position that is on no
    # screen. This is the case a single-rectangle clamp cannot see.
    diagonal = [(0, 0, 1920, 1080), (1920, -1080, 3840, 0)]
    got = _state(x=200, y=-900).clamped(diagonal)
    assert _grabbable(got, diagonal), (
        f"restored to {got.x},{got.y}, which is on neither monitor"
    )


def test_clamped_returns_a_new_object():
    original = _state(alpha=0.0)
    got = original.clamped(ONE_SCREEN)
    assert original.alpha == 0.0, "clamped mutated its input"
    assert got is not original


def test_defaults_survive_their_own_clamp():
    # If the shipped default ever fell outside the guard rails, every fresh
    # install would silently start from a corrected value instead.
    assert uistate.DEFAULT.clamped(ONE_SCREEN) == uistate.DEFAULT


@pytest.mark.parametrize("bad", [[], [(0, 0, 0, 0)], [(0, 0, -5, -5)]])
def test_a_nonsense_desktop_still_yields_a_usable_window(bad):
    # If the desktop rect probe ever returns junk, the clamp must not produce a
    # zero-sized or negative window.
    got = _state().clamped(bad)
    assert got.w >= uistate.MIN_W
    assert got.h >= uistate.MIN_H


# The hand-picked clamp cases above cover the failures we already know about.
# These cover the ones we do not: for ANY saved state and ANY monitor layout,
# the window that comes back has to be usable.

_COORD = st.integers(min_value=-100_000, max_value=100_000)
_ANY_STATE = st.builds(
    UiState,
    w=_COORD, h=_COORD, x=_COORD, y=_COORD,
    font_size=st.integers(min_value=-1000, max_value=1000),
    alpha=st.floats(min_value=-10, max_value=10, allow_nan=False),
)
_ANY_DESKTOP = st.lists(st.tuples(_COORD, _COORD, _COORD, _COORD), max_size=4)


@given(state=_ANY_STATE, desktop=_ANY_DESKTOP)
def test_clamped_output_is_always_usable(state, desktop):
    got = state.clamped(desktop)
    assert got.w >= uistate.MIN_W
    assert got.h >= uistate.MIN_H
    assert uistate.FONT_MIN <= got.font_size <= uistate.FONT_MAX
    assert uistate.ALPHA_MIN <= got.alpha <= uistate.ALPHA_MAX


_A_MONITOR = st.builds(
    lambda left, top, w, h: (left, top, left + w, top + h),
    left=st.integers(-4000, 4000), top=st.integers(-4000, 4000),
    w=st.integers(400, 4000), h=st.integers(300, 3000),
)


@given(state=_ANY_STATE, monitors=st.lists(_A_MONITOR, min_size=1, max_size=3))
def test_clamped_always_lands_on_a_real_monitor(state, monitors):
    # The property that matters: whatever was saved, the window comes back
    # somewhere a mouse can grab it. Checked against each monitor rather than
    # their bounding box, because an L-shaped layout leaves dead space inside
    # the box that belongs to no screen at all.
    got = state.clamped(monitors)
    assert _grabbable(got, monitors), (
        f"{got.x},{got.y} {got.w}x{got.h} is on none of {monitors}"
    )


@given(state=_ANY_STATE, desktop=_ANY_DESKTOP)
def test_clamping_twice_changes_nothing_more(state, desktop):
    # Idempotence. Load applies this, and any later re-clamp of an already
    # clamped value must be a no-op, or a value would drift every restart.
    once = state.clamped(desktop)
    assert once.clamped(desktop) == once
