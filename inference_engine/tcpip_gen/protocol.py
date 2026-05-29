"""Wire-protocol framing for the remote logit sampler.

These helpers are a 1:1 port of the byte layout produced/consumed by the Zig
``tcpip_gen`` client (``inf.zig/src/exe_tcpip_generation.zig``) and the Rust
sampler (``stateful_logit_sampler/.../src/sampler.rs``). Everything is
little-endian.

The functions here are intentionally free of any socket or ``llama_cpp``
dependency so they can be unit-tested without a model or a live server.

Protocol summary (all integers/floats little-endian)::

    Handshake  C->S:  b"HANDSHAKE" + i32(n_vocab) + (piece + b"\\x00") * n_vocab + b"\\x00"
               S->C:  i32(n_vocab)
    Per token  C->S:  b"LOGITS" + n_vocab * f32
               S->C:  i32(count) + i32(token)        # 8 bytes
    End        C->S:  b"\\x00" on EOG, then close

Note on the trailing ``b"\\x00"``: the server detects the end of the vocab
stream by looking for two consecutive null bytes, and its parser emits one
extra ``"NUL"`` token for the first empty piece it sees. Replicating the Zig
sender byte-for-byte means the handshake only balances when exactly one vocab
entry has an empty piece -- this is a property of the original system, not a
bug we introduce here.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence

import numpy as np

#: Magic prefix that opens the handshake message.
HANDSHAKE = b"HANDSHAKE"
#: Magic prefix that opens every per-step logits message.
LOGITS = b"LOGITS"
#: Default TCP host/port the sampler listens on.
HOST = "127.0.0.1"
PORT = 5146

#: Byte length of the fixed handshake header ("HANDSHAKE" + i32 count).
HANDSHAKE_HEADER_LEN = len(HANDSHAKE) + 4
#: Byte length of the handshake response (a single i32 count).
HANDSHAKE_RESPONSE_LEN = 4
#: Byte length of the per-step logits response (i32 count + i32 token).
LOGITS_RESPONSE_LEN = 8


def build_handshake(n_vocab: int, pieces: Sequence[bytes]) -> bytes:
    """Serialize the handshake message.

    Mirrors the Zig writer: the magic string, an ``i32`` token count, every
    token piece followed by a null byte, and finally one extra null byte.

    Args:
        n_vocab: Vocabulary size advertised in the header. Must equal
            ``len(pieces)`` (the Zig client always sends every token).
        pieces: Per-token piece bytes, as returned by ``llama_token_to_piece``
            with ``special=True``.

    Returns:
        The full handshake byte string.
    """
    if len(pieces) != n_vocab:
        raise ValueError(f"piece count {len(pieces)} does not match n_vocab {n_vocab}")
    parts = [HANDSHAKE, struct.pack("<i", n_vocab)]
    for piece in pieces:
        parts.append(piece)
        parts.append(b"\x00")
    parts.append(b"\x00")  # extra trailing null -> double-null terminator
    return b"".join(parts)


def parse_handshake_response(data: bytes) -> int:
    """Parse the server's handshake reply into the echoed token count."""
    if len(data) != HANDSHAKE_RESPONSE_LEN:
        raise ValueError(
            f"handshake response must be {HANDSHAKE_RESPONSE_LEN} bytes, got {len(data)}"
        )
    return struct.unpack("<i", data)[0]


def build_logits_message(logits: Sequence[float] | np.ndarray) -> bytes:
    """Serialize one logits message: ``b"LOGITS"`` + raw little-endian f32s."""
    arr = np.asarray(logits, dtype="<f4")
    return LOGITS + arr.tobytes()


def parse_logits_response(data: bytes) -> tuple[int, int]:
    """Parse the server's per-step reply into ``(count, token_id)``.

    The Rust sampler replies with ``i32(count) + i32(token)`` (8 bytes). This
    deliberately follows the running server rather than the Zig client, which
    mistakenly reads the count as a 64-bit ``usize``.
    """
    if len(data) != LOGITS_RESPONSE_LEN:
        raise ValueError(
            f"logits response must be {LOGITS_RESPONSE_LEN} bytes, got {len(data)}"
        )
    count, token = struct.unpack("<ii", data)
    return count, token
