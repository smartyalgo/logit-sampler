"""llama.cpp model wrapper.

``LlamaModel`` is a thin, faithful port of the llama.cpp C calls used by
``inf.zig/src/exe_tcpip_generation.zig`` (via ``wrapper_llama.zig``). It uses
the low-level ``llama_cpp`` ctypes bindings shipped with ``llama-cpp-python`` so
each method maps almost 1:1 onto the original Zig call.

This module imports ``llama_cpp`` at module load, so it is imported lazily by
the CLI (and never by the protocol/client unit tests).
"""

from __future__ import annotations

import ctypes

import llama_cpp
import numpy as np

#: Buffer size for token-piece detokenization (matches the Zig 128-byte buffer).
_PIECE_BUF_LEN = 128

_backend_ready = False


def _ensure_backend() -> None:
    """Initialize the ggml/llama backend registry exactly once.

    Equivalent to the Zig ``ggml_backend_load_all()`` + ``llama_backend_init``;
    ``llama-cpp-python`` registers the bundled backends when this is called.
    """
    global _backend_ready
    if not _backend_ready:
        llama_cpp.llama_backend_init()
        _backend_ready = True


class LlamaModel:
    """Loads a GGUF model and exposes the calls ``tcpip_gen`` needs."""

    def __init__(self, model_path: str) -> None:
        _ensure_backend()

        model_params = llama_cpp.llama_model_default_params()
        self.model = llama_cpp.llama_model_load_from_file(
            model_path.encode("utf-8"), model_params
        )
        if not self.model:
            raise RuntimeError(f"failed to load model: {model_path}")

        self.vocab = llama_cpp.llama_model_get_vocab(self.model)
        self.n_vocab: int = llama_cpp.llama_vocab_n_tokens(self.vocab)
        self.ctx = None
        # Keep a reference to the live batch token array so ctypes does not free
        # it while ``llama_decode`` is reading through the batch pointer.
        self._batch_tokens: ctypes.Array | None = None

    # -- tokenization / detokenization ---------------------------------------

    def token_to_piece(self, token: int, special: bool) -> bytes:
        """Return the raw bytes for ``token`` (``llama_token_to_piece``)."""
        buf = ctypes.create_string_buffer(_PIECE_BUF_LEN)
        n = llama_cpp.llama_token_to_piece(
            self.vocab, token, buf, _PIECE_BUF_LEN, 0, special
        )
        if n < 0:
            raise RuntimeError(f"llama_token_to_piece failed for token {token}")
        return buf.raw[:n]

    def vocab_pieces(self) -> list[bytes]:
        """Pieces for every token id (``special=True``), for the handshake."""
        return [self.token_to_piece(tid, special=True) for tid in range(self.n_vocab)]

    def tokenize(
        self, text: str, add_special: bool = True, parse_special: bool = True
    ) -> list[int]:
        """Tokenize ``text`` using the two-call sizing pattern from the Zig code."""
        data = text.encode("utf-8")
        # First call with a null buffer returns the negated token count.
        n = -llama_cpp.llama_tokenize(
            self.vocab, data, len(data), None, 0, add_special, parse_special
        )
        if n < 0:
            raise RuntimeError("llama_tokenize sizing call failed")
        tokens = (llama_cpp.llama_token * n)()
        written = llama_cpp.llama_tokenize(
            self.vocab, data, len(data), tokens, n, add_special, parse_special
        )
        if written < 0:
            raise RuntimeError("llama_tokenize failed")
        return list(tokens[:written])

    def is_eog(self, token: int) -> bool:
        """True if ``token`` is an end-of-generation token."""
        return bool(llama_cpp.llama_vocab_is_eog(self.vocab, token))

    def apply_chat_template(
        self, messages: list[dict[str, str]], add_assistant: bool = True
    ) -> str:
        """Render OpenAI-style ``messages`` into a prompt string.

        Uses the model's embedded chat template (``llama_model_chat_template``),
        falling back to the built-in ``chatml`` template when the GGUF carries
        none. ``messages`` is a list of ``{"role": ..., "content": ...}`` dicts;
        ``add_assistant`` appends the assistant role's opening so the model
        continues as the assistant.
        """
        tmpl = llama_cpp.llama_model_chat_template(self.model, None)
        if tmpl is None:
            tmpl = b"chatml"

        n = len(messages)
        arr = (llama_cpp.llama_chat_message * n)()
        # c_char_p only stores the pointer, so keep the encoded bytes alive.
        keepalive: list[bytes] = []
        for i, msg in enumerate(messages):
            role = msg["role"].encode("utf-8")
            content = msg["content"].encode("utf-8")
            keepalive.extend((role, content))
            arr[i].role = role
            arr[i].content = content

        buf_len = 8192
        buf = ctypes.create_string_buffer(buf_len)
        needed = llama_cpp.llama_chat_apply_template(
            tmpl, arr, n, add_assistant, buf, buf_len
        )
        if needed < 0:
            raise RuntimeError("llama_chat_apply_template failed")
        if needed > buf_len:
            buf = ctypes.create_string_buffer(needed)
            needed = llama_cpp.llama_chat_apply_template(
                tmpl, arr, n, add_assistant, buf, needed
            )
            if needed < 0:
                raise RuntimeError("llama_chat_apply_template failed")
        return buf.raw[:needed].decode("utf-8", errors="replace")

    # -- inference context ----------------------------------------------------

    def make_context(self, n_prompt: int, n_predict: int) -> None:
        """Create the inference context, sizing it as the Zig client does."""
        ctx_params = llama_cpp.llama_context_default_params()
        ctx_params.n_ctx = n_prompt + n_predict - 1
        ctx_params.n_batch = n_prompt
        self.ctx = llama_cpp.llama_init_from_model(self.model, ctx_params)
        if not self.ctx:
            raise RuntimeError("failed to create llama context")

    def batch_get_one(self, tokens: list[int]) -> "llama_cpp.llama_batch":
        """Wrap ``tokens`` in a single ``llama_batch`` (``llama_batch_get_one``)."""
        arr = (llama_cpp.llama_token * len(tokens))(*tokens)
        self._batch_tokens = arr  # keep alive for the duration of decode
        return llama_cpp.llama_batch_get_one(arr, len(tokens))

    def decode(self, batch: "llama_cpp.llama_batch") -> int:
        """Run a forward pass; returns the ``llama_decode`` status (0 == ok)."""
        if self.ctx is None:
            raise RuntimeError("decode() called before make_context()")
        return llama_cpp.llama_decode(self.ctx, batch)

    def get_last_logits(self) -> np.ndarray:
        """Return the logits for the last decoded position as ``n_vocab`` f32s.

        Uses ``llama_get_logits_ith(ctx, -1)`` -- the last position -- which is
        the correct row for autoregressive generation (the Zig client read row
        0, which only works for single-token batches).
        """
        if self.ctx is None:
            raise RuntimeError("get_last_logits() called before make_context()")
        ptr = llama_cpp.llama_get_logits_ith(self.ctx, -1)
        if not ptr:
            raise RuntimeError("llama_get_logits_ith returned NULL")
        return np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).astype(
            "<f4", copy=True
        )

    # -- cleanup --------------------------------------------------------------

    def free_context(self) -> None:
        """Free just the inference context, keeping the model loaded.

        Lets a long-lived server create a fresh context per request without
        reloading the model.
        """
        if self.ctx is not None:
            llama_cpp.llama_free(self.ctx)
            self.ctx = None
        self._batch_tokens = None

    def close(self) -> None:
        """Free the context and model."""
        self.free_context()
        if self.model is not None:
            llama_cpp.llama_model_free(self.model)
            self.model = None

    def __enter__(self) -> "LlamaModel":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
