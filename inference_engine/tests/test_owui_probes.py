"""Slow regression test: replays Open WebUI's probe set against a real server.

Off by default. Run with::

    RUN_LIVE=1 TCPIP_GEN_MODEL=/path/to/model.gguf uv run pytest tests/test_owui_probes.py -v

What it does
------------
Boots the real ``tcpip_gen.server`` in a subprocess on a free port (needs the
GGUF model and the Rust sampler on ``127.0.0.1:5146``), then issues the exact
URLs Open WebUI hit in the failure logs::

    /api/version    /v1/api/version
    /api/tags       /v1/api/tags
    /api/ps         /v1/api/ps
    /v1/models

Asserts each returns 200 with a well-formed body. This turns the "configure
Open WebUI and click around" verification into a deterministic regression test
for the original 404/500.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

RUN_LIVE = os.environ.get("RUN_LIVE") == "1"
MODEL_PATH = os.environ.get("TCPIP_GEN_MODEL", "")
SAMPLER_HOST = os.environ.get("TCPIP_GEN_SAMPLER_HOST", "127.0.0.1")
SAMPLER_PORT = int(os.environ.get("TCPIP_GEN_SAMPLER_PORT", "5146"))
STARTUP_TIMEOUT_SECONDS = 120


def _sampler_listening() -> bool:
    try:
        with socket.create_connection((SAMPLER_HOST, SAMPLER_PORT), timeout=0.5):
            return True
    except OSError:
        return False


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(not RUN_LIVE, reason="set RUN_LIVE=1 to enable")
@pytest.mark.skipif(not MODEL_PATH, reason="set TCPIP_GEN_MODEL to a GGUF path")
@pytest.mark.skipif(
    not _sampler_listening(),
    reason=f"sampler not listening on {SAMPLER_HOST}:{SAMPLER_PORT}",
)
def test_open_webui_probe_set_returns_200():
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tcpip_gen.server",
            "--model",
            MODEL_PATH,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--sampler-host",
            SAMPLER_HOST,
            "--sampler-port",
            str(SAMPLER_PORT),
            "--max-tokens",
            "16",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"

    try:
        _wait_for_ready(proc, base)

        # The exact paths Open WebUI hit in the original failure logs, plus the
        # OpenAI listing for completeness.
        probes = [
            "/v1/api/version",
            "/v1/api/tags",
            "/v1/api/ps",
            "/api/version",
            "/api/tags",
            "/api/ps",
            "/v1/models",
        ]
        for path in probes:
            resp = httpx.get(f"{base}{path}", timeout=5.0)
            assert resp.status_code == 200, f"{path} -> {resp.status_code}"
            body = resp.json()
            if path.endswith("/version"):
                version = body["version"]
                parts = version.split(".")
                assert all(p.isdigit() for p in parts), f"{path} version={version!r}"
            elif path.endswith(("/tags", "/ps")):
                assert isinstance(body["models"], list) and body["models"], path
            elif path == "/v1/models":
                assert body["object"] == "list" and body["data"], path
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _wait_for_ready(proc: subprocess.Popen, base: str) -> None:
    deadline = time.time() + STARTUP_TIMEOUT_SECONDS
    while time.time() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"server subprocess exited early with code {proc.returncode}")
        try:
            r = httpx.get(f"{base}/v1/models", timeout=1.0)
            if r.status_code == 200:
                return
        except httpx.RequestError:
            pass
        time.sleep(0.5)
    pytest.fail(f"server did not become ready within {STARTUP_TIMEOUT_SECONDS}s")
