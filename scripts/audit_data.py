"""Validate a manifest without printing clinical text or patient identifiers."""
import argparse
import json
from ocuweave.data.auditing.manifest import audit_manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--data-root")
    args = parser.parse_args()
    try:
        report = audit_manifest(args.manifest, check_files=args.check_files, data_root=args.data_root)
    except (ValueError, TypeError, OSError) as error:
        parser.exit(2, f"Audit failed ({type(error).__name__}); review the local manifest.\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
