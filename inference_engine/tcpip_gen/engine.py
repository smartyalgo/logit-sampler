"""The generation engine behind the OpenAI-compatible server.

The HTTP layer (:mod:`tcpip_gen.server`) depends only on the :class:`Engine`
protocol, so it can be tested with a fake. :class:`LlamaSamplerEngine` is the
real implementation: it renders a chat prompt, runs the shared
:func:`tcpip_gen.generation.generate_tokens` loop (decode locally, sample
remotely), and turns the token stream into UTF-8-safe text deltas while applying
``max_tokens`` and ``stop`` locally.

An ``Engine.chat(...)`` call returns an iterator of assistant text deltas; the
iterator's *return value* (``StopIteration.value``) is a
:class:`GenerationResult` with the finish reason and token counts.
"""

from __future__ import annotations

import codecs
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .generation import generate_tokens

if TYPE_CHECKING:
    # Type-only: avoids importing llama_cpp at runtime (kept out of the HTTP
    # layer and its tests). `from __future__ import annotations` makes the
    # annotations below lazy strings, so these names are never evaluated here.
    from .client import SamplerClient
    from .llama_backend import LlamaModel


@dataclass
class GenerationResult:
    """Returned via ``StopIteration.value`` when an ``Engine.chat`` stream ends."""

    finish_reason: str  # "stop" (EOG or stop-string) | "length" (max_tokens)
    prompt_tokens: int
    completion_tokens: int


@runtime_checkable
class Engine(Protocol):
    """What the HTTP server needs from a generation backend."""

    model_id: str

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        stop: Sequence[str],
    ) -> Iterator[str]:
        """Yield assistant text deltas; return a :class:`GenerationResult`."""
        ...


class _Utf8Stream:
    """Incremental UTF-8 decoder; never splits a multi-byte character."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def push(self, data: bytes) -> str:
        return self._decoder.decode(data)

    def flush(self) -> str:
        return self._decoder.decode(b"", final=True)


class _StopBuffer:
    """Streams text while holding back any tail that might begin a stop string.

    Withholding the last ``max_stop_len - 1`` characters guarantees we never emit
    a fragment that later turns out to be part of a stop sequence.
    """

    def __init__(self, stops: Sequence[str]) -> None:
        self._stops = [s for s in stops if s]
        self._max = max((len(s) for s in self._stops), default=0)
        self._text = ""
        self._emitted = 0
        self.hit = False

    def push(self, text: str) -> str:
        self._text += text
        if self._stops:
            idx = self._earliest_stop()
            if idx is not None:
                self.hit = True
                return self._take(idx)
            cut = max(self._emitted, len(self._text) - (self._max - 1))
        else:
            cut = len(self._text)
        return self._take(cut)

    def flush(self) -> str:
        """Emit any withheld tail (nothing if a stop string was hit)."""
        return "" if self.hit else self._take(len(self._text))

    def _take(self, upto: int) -> str:
        out = self._text[self._emitted : upto]
        self._emitted = max(self._emitted, upto)
        return out

    def _earliest_stop(self) -> int | None:
        best: int | None = None
        for s in self._stops:
            i = self._text.find(s)
            if i != -1 and (best is None or i < best):
                best = i
        return best


class LlamaSamplerEngine:
    """Engine backed by a local llama.cpp model and the remote logit sampler.

    Owns the single model + sampler connection and serializes generation with a
    lock (the llama.cpp context and the socket are not concurrency-safe). The
    sampler connection is reused across requests — the inter-token ``finish()``
    null byte is never sent here, since it would desync the sampler's framing.
    """

    def __init__(self, model: LlamaModel, client: SamplerClient, model_id: str) -> None:
        self._model = model
        self._client = client
        self.model_id = model_id
        self._lock = threading.Lock()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        stop: Sequence[str],
    ) -> Iterator[str]:
        max_tokens = max(1, max_tokens)
        with self._lock:
            model = self._model
            prompt = model.apply_chat_template(messages)
            prompt_tokens = model.tokenize(prompt)
            model.make_context(len(prompt_tokens), max_tokens)

            utf8 = _Utf8Stream()
            stops = _StopBuffer(stop)
            completion_tokens = 0
            finish_reason = "length"
            try:
                tokens = generate_tokens(model, self._client, prompt_tokens, max_tokens)
                try:
                    while True:
                        token = next(tokens)
                        completion_tokens += 1
                        piece = model.token_to_piece(token, special=False)
                        delta = stops.push(utf8.push(piece))
                        if delta:
                            yield delta
                        if stops.hit:
                            finish_reason = "stop"
                            break
                except StopIteration as done:
                    finish_reason = done.value
                    tail = stops.push(utf8.flush())
                    if tail:
                        yield tail
                    remainder = stops.flush()
                    if remainder:
                        yield remainder
            finally:
                model.free_context()

            return GenerationResult(
                finish_reason=finish_reason,
                prompt_tokens=len(prompt_tokens),
                completion_tokens=completion_tokens,
            )
