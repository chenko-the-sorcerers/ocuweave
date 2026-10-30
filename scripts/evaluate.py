"""Evaluate a frozen checkpoint once on its audited held-out split."""
import argparse
import json
from ocuweave.training.engine import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'archive', 'checkpoint', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--split', choices=['validation', 'test'], default='test')
    parser.add_argument('--allow-unverified-patients', action='store_true')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    result = evaluate(args.manifest, args.archive, args.checkpoint, args.output,
        split=args.split, allow_unverified=args.allow_unverified_patients, device=args.device)
    print(json.dumps(result['metrics'], indent=2))


if __name__ == '__main__':
    main()
