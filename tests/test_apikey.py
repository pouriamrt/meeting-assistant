"""Key storage. A credential at rest, so the failure modes matter more than usual."""

import pytest

from assistant import apikey

KEY = "sk-ant-api03-not-a-real-key-000000"


@pytest.fixture(autouse=True)
def isolate_key(tmp_path, monkeypatch):
    """Never touch the real key.bin. Same reason as the ui.json fixture: the
    developer's own credential lives at that path."""
    monkeypatch.setattr(apikey, "_key_path", lambda: tmp_path / "key.bin")


def test_environment_wins_over_a_stored_key(monkeypatch):
    # A shell export has to beat whatever a previous run saved, or a key can
    # only be changed by finding and deleting a binary file.
    apikey.save("sk-ant-stored-one")
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    assert apikey.resolve() == KEY


def test_stored_key_is_used_when_the_environment_is_empty(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    apikey.save(KEY)
    assert apikey.resolve() == KEY


def test_round_trip_through_dpapi():
    apikey.save(KEY)
    assert apikey.stored() == KEY


def test_the_key_is_not_written_in_the_clear():
    # The whole reason for DPAPI. A plaintext file next to ui.json would be
    # readable by every process running as this user.
    apikey.save(KEY)
    raw = apikey._key_path().read_bytes()
    assert KEY.encode() not in raw
    assert b"sk-ant" not in raw


def test_entropy_is_required_to_decrypt(monkeypatch):
    # DPAPI's entropy is its third parameter; the second is a description
    # string that changes nothing. Passing the salt in the wrong slot encrypts
    # fine and protects nothing, so prove the salt actually participates.
    apikey.save(KEY)
    blob = apikey._key_path().read_bytes()
    monkeypatch.setattr(apikey, "_ENTROPY", b"some-other-app")
    with pytest.raises(OSError):
        apikey._dpapi(apikey._crypt32.CryptUnprotectData, blob)


def test_missing_file_is_not_an_error():
    assert apikey.stored() is None


def test_a_corrupt_blob_asks_again_instead_of_crashing(caplog):
    # A blob written by a different Windows account looks exactly like this.
    # Refusing to start would leave no way to fix it from inside the app.
    path = apikey._key_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a dpapi blob at all")
    with caplog.at_level("WARNING"):
        assert apikey.stored() is None


def test_a_failed_save_does_not_raise(monkeypatch):
    # Losing the convenience of a remembered key is survivable. Taking the
    # launch down over it is not.
    def boom(*a, **k):
        raise OSError("read-only volume")

    monkeypatch.setattr(apikey.Path, "write_bytes", boom)
    apikey.save(KEY)          # must not raise


def test_nothing_key_shaped_reaches_the_log(monkeypatch, caplog):
    # Every logging call in this module passes exc_info, so the record carries
    # a traceback, and a traceback carries frame locals - which is exactly
    # where a key would hide. Format the records in full rather than reading
    # getMessage(), which omits the traceback and would pass either way.
    import logging

    path = apikey._key_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with caplog.at_level("DEBUG"):
        path.write_bytes(b"garbage")     # unreadable blob, logs a warning
        apikey.stored()

        def boom(*a, **k):
            raise OSError("nope")

        # Patched only now: it would otherwise break the write above too.
        monkeypatch.setattr(apikey.Path, "write_bytes", boom)
        apikey.save(KEY)                 # failed write, logs a warning

    fmt = logging.Formatter("%(message)s")
    rendered = "\n".join(fmt.format(r) for r in caplog.records)
    assert caplog.records, "nothing was logged, so this proves nothing"
    assert "Traceback" in rendered, "no traceback captured, the weak case again"
    assert KEY not in rendered
    assert "sk-ant" not in rendered


def test_resolve_raises_when_the_prompt_is_dismissed(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(apikey, "prompt", lambda: None)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        apikey.resolve()


def test_resolve_saves_what_the_prompt_returned(monkeypatch):
    # Otherwise it asks again on every single launch.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(apikey, "prompt", lambda: KEY)
    assert apikey.resolve() == KEY
    assert apikey.stored() == KEY, "the key was used but never persisted"
