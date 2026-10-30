"""Download a pinned, legally open OTFID release; no credentials or paywall bypass."""
import argparse
from pathlib import Path
from ocuweave.data.acquisition import acquire_otfid

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(acquire_otfid(args.output))
