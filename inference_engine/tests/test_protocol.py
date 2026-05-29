"""Unit tests for the pure wire-protocol framing helpers."""

import struct

import numpy as np
import pytest

from tcpip_gen import protocol


def test_build_handshake_layout():
    pieces = [b"a", b"bb", b"ccc"]
    msg = protocol.build_handshake(3, pieces)

    assert msg[:9] == b"HANDSHAKE"
    assert struct.unpack("<i", msg[9:13])[0] == 3
    # Each piece followed by a null, then one extra trailing null.
    assert msg[13:] == b"a\x00bb\x00ccc\x00\x00"


def test_build_handshake_empty_piece_creates_double_null():
    # An empty final piece + the trailing null yields the double-null the
    # server uses to detect end-of-vocab, and balances its NUL-token quirk.
    msg = protocol.build_handshake(2, [b"x", b""])
    assert msg[13:] == b"x\x00\x00\x00"
    assert msg.endswith(b"\x00\x00")


def test_build_handshake_rejects_count_mismatch():
    with pytest.raises(ValueError):
        protocol.build_handshake(3, [b"a", b"b"])


def test_parse_handshake_response():
    assert protocol.parse_handshake_response(struct.pack("<i", 128256)) == 128256


def test_parse_handshake_response_bad_length():
    with pytest.raises(ValueError):
        protocol.parse_handshake_response(b"\x00\x00\x00")


def test_build_logits_message_layout():
    logits = [1.0, 2.0, -3.5, 0.0]
    msg = protocol.build_logits_message(logits)

    assert msg[:6] == b"LOGITS"
    assert len(msg) == 6 + 4 * len(logits)
    np.testing.assert_array_equal(
        np.frombuffer(msg[6:], dtype="<f4"),
        np.asarray(logits, dtype="<f4"),
    )


def test_build_logits_message_accepts_numpy():
    arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    msg = protocol.build_logits_message(arr)
    assert msg[:6] == b"LOGITS"
    np.testing.assert_array_equal(np.frombuffer(msg[6:], dtype="<f4"), arr)


def test_parse_logits_response_roundtrip():
    data = struct.pack("<ii", 128256, 42)
    assert protocol.parse_logits_response(data) == (128256, 42)


def test_parse_logits_response_bad_length():
    with pytest.raises(ValueError):
        protocol.parse_logits_response(struct.pack("<i", 5))  # only 4 bytes
