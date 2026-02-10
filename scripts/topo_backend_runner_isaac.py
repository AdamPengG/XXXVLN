#!/usr/bin/env python
from __future__ import annotations

import sys

import topo_backend_runner as runner


def main() -> None:
    if "--backend" not in sys.argv:
        sys.argv.extend(["--backend", "isaac"])
    runner.main()


if __name__ == "__main__":
    main()
