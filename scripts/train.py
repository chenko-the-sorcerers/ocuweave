"""Train an image-only segmentation experiment with validation-only selection."""
import argparse
import json
from pathlib import Path
from ocuweave.training.engine import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', required=True, help='JSON experiment configuration')
    parser.add_argument('--allow-unverified-patients', action='store_true')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    checkpoint = train(args.manifest, args.archive, args.output,
        json.loads(Path(args.config).read_text()),
        allow_unverified=args.allow_unverified_patients, device=args.device)
    print(checkpoint)


if __name__ == '__main__':
    main()
