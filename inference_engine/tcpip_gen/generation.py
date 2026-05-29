"""The core decode -> remote-sample -> next-token loop.

Extracted so both the CLI (:mod:`tcpip_gen.__main__`) and the HTTP server
(:mod:`tcpip_gen.engine`) share one implementation.

The generator yields sampled token ids. Its *return value* (available as
``StopIteration.value``) is the finish reason:

* ``"stop"``   -- generation hit an end-of-generation token.
* ``"length"`` -- generation hit the ``n_predict`` cap.

It deliberately does **not** send the sampler's end-of-generation null byte
(``SamplerClient.finish``): a one-shot CLI run wants that byte before closing,
but a long-lived server reuses the connection and must not inject it (it would
desync the sampler's fixed-length framing). The caller decides.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import SamplerClient
    from .llama_backend import LlamaModel


def generate_tokens(
    model: LlamaModel,
    client: SamplerClient,
    prompt_tokens: list[int],
    n_predict: int,
) -> Iterator[int]:
    """Autoregressively generate up to ``n_predict`` tokens.

    The caller must have created the inference context already
    (``model.make_context``) and completed the sampler handshake
    (``client.handshake``). Yields each sampled token id; returns the finish
    reason (``"stop"`` or ``"length"``) via ``StopIteration.value``.
    """
    n_prompt = len(prompt_tokens)
    batch = model.batch_get_one(prompt_tokens)
    n_pos = 0
    while n_pos + batch.n_tokens < n_prompt + n_predict:
        if model.decode(batch) != 0:
            raise RuntimeError("llama_decode failed during generation")
        n_pos += batch.n_tokens

        logits = model.get_last_logits()
        token = client.sample(logits)

        if model.is_eog(token):
            return "stop"

        yield token
        batch = model.batch_get_one([token])

    return "length"
