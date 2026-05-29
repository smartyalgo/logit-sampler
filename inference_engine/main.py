"""Convenience shim so ``python main.py ...`` runs the tcpip_gen CLI.

Prefer ``python -m tcpip_gen <model.gguf> "<prompt>"``; this just forwards to it.
"""

from tcpip_gen.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
