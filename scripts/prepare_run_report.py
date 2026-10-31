"""Create the nine-run report without registering a data contract or running training."""
import argparse
from pathlib import Path
from ocuweave.training.ablation import configurations, summarize

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True,exist_ok=False)
    summarize(root,configurations())
    print('Nine pending rows written; no manifest registered and no training performed.')
