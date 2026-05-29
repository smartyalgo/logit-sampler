"""Optional interop test against the real sampler on 127.0.0.1:5146.

Skipped automatically when nothing is listening on the port. It performs a
real handshake with a tiny synthetic vocab -- no model is loaded.

The vocab is crafted so the server's parser yields exactly ``n_vocab`` tokens:
the server emits one extra ``"NUL"`` token for the first empty piece, and the
trailing null otherwise adds a stray token. Including exactly one empty piece
balances that quirk (see :mod:`tcpip_gen.protocol`).
"""

import socket

import pytest

from tcpip_gen.client import SamplerClient
from tcpip_gen.protocol import HOST, PORT


def _sampler_listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


@pytest.mark.skipif(
    not _sampler_listening(HOST, PORT),
    reason=f"no sampler listening on {HOST}:{PORT}",
)
def test_real_handshake_echoes_token_count():
    # 4 entries, exactly one empty -> server parses ["a","b","c","NUL"] == 4.
    pieces = [b"a", b"b", b"c", b""]
    with SamplerClient(HOST, PORT) as client:
        echoed = client.handshake(len(pieces), pieces)
        assert echoed == len(pieces)
