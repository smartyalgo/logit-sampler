"""TCP client for the remote logit sampler.

``SamplerClient`` owns the socket and drives the protocol defined in
:mod:`tcpip_gen.protocol`: it performs the handshake, then exchanges one
``LOGITS`` message per generated token. It has no knowledge of llama.cpp -- the
caller supplies the vocab pieces and the per-step logits.
"""

from __future__ import annotations

import socket
from collections.abc import Sequence

import numpy as np

from . import protocol


class SamplerError(RuntimeError):
    """Raised when the sampler violates the expected protocol."""


class SamplerClient:
    """A thin, synchronous client for the remote logit sampler."""

    def __init__(self, host: str = protocol.HOST, port: int = protocol.PORT) -> None:
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None
        self._n_vocab: int | None = None

    # -- connection lifecycle -------------------------------------------------

    def connect(self, timeout: float | None = None) -> None:
        """Open the TCP connection to the sampler."""
        self._sock = socket.create_connection((self.host, self.port), timeout=timeout)
        # Disable Nagle so each small framed message goes out promptly.
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def close(self) -> None:
        """Close the connection if open."""
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> SamplerClient:
        if self._sock is None:
            self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- protocol -------------------------------------------------------------

    def handshake(self, n_vocab: int, pieces: Sequence[bytes]) -> int:
        """Send the vocabulary and verify the echoed token count.

        Args:
            n_vocab: Vocabulary size.
            pieces: Per-token piece bytes (``special=True``).

        Returns:
            The token count echoed by the server (equal to ``n_vocab``).

        Raises:
            SamplerError: If the server echoes a different count.
        """
        self._send(protocol.build_handshake(n_vocab, pieces))
        echoed = protocol.parse_handshake_response(
            self._recv_exact(protocol.HANDSHAKE_RESPONSE_LEN)
        )
        if echoed != n_vocab:
            raise SamplerError(f"sampler echoed {echoed} tokens, expected {n_vocab}")
        self._n_vocab = n_vocab
        return echoed

    def sample(self, logits: Sequence[float] | np.ndarray) -> int:
        """Send one step of logits and return the sampled token id.

        Raises:
            SamplerError: If called before :meth:`handshake`, or if the server
                reports consuming a different number of logits than expected.
        """
        if self._n_vocab is None:
            raise SamplerError("sample() called before handshake()")
        self._send(protocol.build_logits_message(logits))
        count, token = protocol.parse_logits_response(
            self._recv_exact(protocol.LOGITS_RESPONSE_LEN)
        )
        if count != self._n_vocab:
            raise SamplerError(
                f"sampler consumed {count} logits, expected {self._n_vocab}"
            )
        return token

    def finish(self) -> None:
        """Signal end-of-generation to the server (a single null byte).

        Mirrors the Zig client, which writes ``\\x00`` when it hits an EOG
        token. The server also treats a closed socket as completion.
        """
        self._send(b"\x00")

    # -- socket helpers -------------------------------------------------------

    def _send(self, data: bytes) -> None:
        if self._sock is None:
            raise SamplerError("client is not connected")
        self._sock.sendall(data)

    def _recv_exact(self, n: int) -> bytes:
        """Read exactly ``n`` bytes or raise if the peer closes early."""
        if self._sock is None:
            raise SamplerError("client is not connected")
        chunks: list[bytes] = []
        remaining = n
        while remaining > 0:
            chunk = self._sock.recv(remaining)
            if not chunk:
                raise SamplerError(
                    f"connection closed with {remaining} of {n} bytes unread"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
