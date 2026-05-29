"""Tests for the engine's UTF-8 streaming and stop-string helpers (no llama_cpp)."""

from tcpip_gen.engine import _StopBuffer, _Utf8Stream


def test_utf8_stream_never_splits_multibyte():
    s = _Utf8Stream()
    # "é" is two bytes (0xC3 0xA9); the first byte alone must emit nothing.
    assert s.push(b"\xc3") == ""
    assert s.push(b"\xa9") == "é"
    assert s.push(b"ok") == "ok"
    assert s.flush() == ""


def test_utf8_stream_flush_replaces_incomplete_tail():
    s = _Utf8Stream()
    assert s.push(b"\xc3") == ""  # dangling lead byte
    assert s.flush() == "�"  # replacement char on final flush


def test_stopbuffer_passes_text_through_without_stops():
    b = _StopBuffer([])
    out = b.push("abc") + b.push("def") + b.flush()
    assert out == "abcdef"
    assert b.hit is False


def test_stopbuffer_truncates_at_stop_string():
    b = _StopBuffer(["END"])
    out = b.push("hello ") + b.push("wor") + b.push("ld END more") + b.flush()
    assert b.hit is True
    assert out == "hello world "  # everything before the stop, nothing after


def test_stopbuffer_holds_back_partial_stop_prefix():
    # A trailing fragment that could begin a stop string is withheld until resolved.
    b = _StopBuffer(["STOP"])
    emitted = b.push("abcST")  # "ST" could start "STOP" -> hold it back
    assert "ST" not in emitted
    rest = b.push("X") + b.flush()  # "STX" is not a stop -> everything flushes
    assert emitted + rest == "abcSTX"
    assert b.hit is False
