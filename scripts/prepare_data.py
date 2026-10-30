"""Validate and copy an existing manifest; never synthesize clinical records or splits."""
import argparse
import json
from pathlib import Path
from ocuweave.data.auditing.manifest import audit_manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit_manifest(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        json.dump(json.loads(args.manifest.read_text()), destination, indent=2)
        destination.write("\n")
    print("Validated manifest written. No records were generated or relabeled.")

if __name__ == "__main__": main()
