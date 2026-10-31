"""Compatibility launcher for validation-first installed CLI evaluation."""

import sys
from ocuweave.cli import main

if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])
