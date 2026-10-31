"""Inspect numeric images and development-note flags without unpickling objects."""
import argparse
import json
from ocuweave.data.auditing.fairvlmed_images import audit_images

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
print(json.dumps(audit_images(args.root, args.output), indent=2))
