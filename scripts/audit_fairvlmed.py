"""Audit FairVLMed filename pairing and clinical-table structure without printing notes."""
import argparse
import json
from ocuweave.data.adapters.fairvlmed import audit_fairvlmed

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
print(json.dumps(audit_fairvlmed(args.root,args.output),indent=2))
