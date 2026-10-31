"""Allow python -m ocuweave with the same interface as the installed command."""
from ocuweave.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
