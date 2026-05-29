"""Tests for the shared generate_tokens loop using fake model/client objects."""

import numpy as np

from tcpip_gen.generation import generate_tokens


class _FakeBatch:
    def __init__(self, n_tokens: int) -> None:
        self.n_tokens = n_tokens


class _FakeModel:
    """Minimal stand-in for LlamaModel (no llama_cpp)."""

    def __init__(self, eog_token: int, n_vocab: int = 8) -> None:
        self.eog = eog_token
        self.n_vocab = n_vocab
        self.decode_calls = 0

    def batch_get_one(self, tokens):
        return _FakeBatch(len(tokens))

    def decode(self, batch):
        self.decode_calls += 1
        return 0

    def get_last_logits(self):
        return np.zeros(self.n_vocab, dtype="<f4")

    def is_eog(self, token):
        return token == self.eog


class _FakeClient:
    def __init__(self, tokens):
        self._tokens = list(tokens)
        self._i = 0
        self.finish_called = False

    def sample(self, logits):
        token = self._tokens[self._i]
        self._i += 1
        return token

    def finish(self):
        self.finish_called = True


def _run(model, client, prompt, n_predict):
    out: list[int] = []
    gen = generate_tokens(model, client, prompt, n_predict)
    try:
        while True:
            out.append(next(gen))
    except StopIteration as done:
        return out, done.value


def test_stops_on_eog_without_sending_finish():
    model = _FakeModel(eog_token=99)
    client = _FakeClient([1, 2, 3, 99, 4])
    out, reason = _run(model, client, prompt=[5, 6], n_predict=10)
    assert out == [1, 2, 3]
    assert reason == "stop"
    # generate_tokens must never send the inter-request finish() byte.
    assert client.finish_called is False


def test_caps_at_n_predict():
    model = _FakeModel(eog_token=99)
    client = _FakeClient([1, 2, 3, 4, 5, 6])
    # prompt of 2 tokens + n_predict 3 -> stops after 3 generated tokens.
    out, reason = _run(model, client, prompt=[5, 6], n_predict=3)
    assert out == [1, 2, 3]
    assert reason == "length"
    assert client.finish_called is False
