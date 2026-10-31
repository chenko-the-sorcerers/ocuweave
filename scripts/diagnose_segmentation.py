"""Analyze a frozen checkpoint on training/validation; test is explicitly rejected."""
import argparse
import json
import torch
from ocuweave.evaluation.diagnostics import diagnose

parser = argparse.ArgumentParser(description=__doc__)
for name in ("manifest", "archive", "checkpoint", "output"):
    parser.add_argument("--" + name, required=True)
parser.add_argument("--split", choices=("train", "validation"), default="validation")
parser.add_argument("--device", default="cpu")
parser.add_argument("--allow-unverified-patients", action="store_true")
args = parser.parse_args()
torch.set_num_threads(2)
print(json.dumps(diagnose(args.manifest, args.archive, args.checkpoint, args.output,
      split=args.split, device=args.device, allow_unverified=args.allow_unverified_patients), indent=2))
