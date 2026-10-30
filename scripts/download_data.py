"""Download official OTFID or FairVLMed files with provenance and checksum checks."""
import argparse
from pathlib import Path
from ocuweave.data.acquisition import acquire_otfid, acquire_fairvlmed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', choices=['otfid','fairvlmed'], default='otfid')
    args = parser.parse_args()
    print({'otfid':acquire_otfid,'fairvlmed':acquire_fairvlmed}[args.dataset](args.output))
