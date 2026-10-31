"""Create a private, CPU-only development photo-mask and resize review bundle."""
import argparse
import json
from ocuweave.data.auditing.review_bundle import build_review

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'archive', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--allow-unverified-patients', action='store_true')
    args = parser.parse_args()
    report = build_review(args.manifest, args.archive, args.output,
                          allow_unverified=args.allow_unverified_patients)
    print(json.dumps({k:v for k,v in report.items() if k!='pair_failures'}, indent=2))
