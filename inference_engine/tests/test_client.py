"""Unit tests for SamplerClient against an in-process fake sampler."""

import pytest

from tcpip_gen.client import SamplerClient, SamplerError

from .fake_sampler import FakeSampler


def test_handshake_and_sample_sequence():
    n_vocab = 5
    pieces = [b"a", b"b", b"c", b"d", b"e"]
    with FakeSampler(tokens=[3, 4, 2]) as server:
        with SamplerClient("127.0.0.1", server.port) as client:
            assert client.handshake(n_vocab, pieces) == n_vocab
            assert client.sample([0.0] * n_vocab) == 3
            assert client.sample([0.1, 0.2, 0.3, 0.4, 0.5]) == 4
            assert client.sample([1.0] * n_vocab) == 2
            client.finish()
    assert server.error is None


def test_handshake_count_mismatch_raises():
    with FakeSampler(tokens=[], handshake_count=7) as server:
        with SamplerClient("127.0.0.1", server.port) as client:
            with pytest.raises(SamplerError, match="echoed 7"):
                client.handshake(5, [b"a", b"b", b"c", b"d", b"e"])


def test_sample_logit_count_mismatch_raises():
    n_vocab = 4
    with FakeSampler(tokens=[1], logits_count=99) as server:
        with SamplerClient("127.0.0.1", server.port) as client:
            client.handshake(n_vocab, [b"a", b"b", b"c", b"d"])
            with pytest.raises(SamplerError, match="consumed 99"):
                client.sample([0.0] * n_vocab)


def test_sample_before_handshake_raises():
    with FakeSampler(tokens=[]) as server:
        with SamplerClient("127.0.0.1", server.port) as client:
            with pytest.raises(SamplerError, match="before handshake"):
                client.sample([0.0, 0.0])


def test_send_without_connect_raises():
    client = SamplerClient("127.0.0.1", 1)
    with pytest.raises(SamplerError, match="not connected"):
        client.handshake(1, [b"a"])
