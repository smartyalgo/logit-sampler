"""
``tcpip_gen`` runs llama.cpp inference locally but delegates token *sampling* to
a remote sampler over TCP. This package mirrors that client:

* :mod:`tcpip_gen.protocol` -- pure byte framing (no sockets, no llama.cpp).
* :mod:`tcpip_gen.client` -- :class:`~tcpip_gen.client.SamplerClient` TCP driver.
* :mod:`tcpip_gen.llama_backend` -- :class:`~tcpip_gen.llama_backend.LlamaModel`
  (imports ``llama_cpp``; loaded lazily by the CLI).

The CLI lives in :mod:`tcpip_gen.__main__` (``python -m tcpip_gen``).
"""

from . import client, protocol
from .client import SamplerClient, SamplerError

__all__ = ["protocol", "client", "SamplerClient", "SamplerError"]
