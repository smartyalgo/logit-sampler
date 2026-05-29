"""An in-process TCP server that imitates the Rust sampler's framing.

Used by the client unit tests. It mirrors the *wire* behaviour of the real
sampler (4-byte handshake echo, 8-byte ``<ii`` per-step reply) but does not
reproduce the server's quirky vocab parser -- it simply echoes the token count
read from the handshake header, which is enough to exercise the client.
"""

from __future__ import annotations

import socket
import struct
import threading


class FakeSampler:
    """Context manager running a one-connection fake sampler on a random port.

    Args:
        tokens: Token ids to return, one per ``sample`` call.
        handshake_count: If set, echo this instead of the count from the header.
        logits_count: If set, put this in the per-step reply instead of the
            handshake count (to exercise the client's mismatch check).
    """

    def __init__(
        self,
        tokens: list[int],
        handshake_count: int | None = None,
        logits_count: int | None = None,
    ) -> None:
        self.tokens = tokens
        self.handshake_count = handshake_count
        self.logits_count = logits_count
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(1)
        self.port: int = self._srv.getsockname()[1]
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self) -> "FakeSampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._thread.join(timeout=2)
        self._srv.close()

    def _serve(self) -> None:
        try:
            conn, _ = self._srv.accept()
            with conn:
                n_vocab = self._read_handshake(conn)
                echo = (
                    self.handshake_count
                    if self.handshake_count is not None
                    else n_vocab
                )
                conn.sendall(struct.pack("<i", echo))

                step_count = (
                    self.logits_count if self.logits_count is not None else n_vocab
                )
                for tok in self.tokens:
                    self._read_logits(conn, n_vocab)
                    conn.sendall(struct.pack("<ii", step_count, tok))
        except BaseException as exc:  # noqa: BLE001 - surfaced to the test thread
            self.error = exc

    @staticmethod
    def _read_handshake(conn: socket.socket) -> int:
        header = _recv_exact(conn, 13)
        assert header[:9] == b"HANDSHAKE"
        n_vocab = struct.unpack("<i", header[9:13])[0]
        # Drain the vocab stream up to the double-null terminator.
        body = b""
        while not (len(body) >= 2 and body[-1] == 0 and body[-2] == 0):
            chunk = conn.recv(4096)
            if not chunk:
                break
            body += chunk
        return n_vocab

    @staticmethod
    def _read_logits(conn: socket.socket, n_vocab: int) -> None:
        data = _recv_exact(conn, 6 + 4 * n_vocab)
        assert data[:6] == b"LOGITS"


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            raise AssertionError(f"peer closed with {remaining}/{n} bytes unread")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
