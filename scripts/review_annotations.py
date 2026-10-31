"""Inspect frozen training/validation geometry without altering annotations."""
import argparse
import json
from ocuweave.data.auditing.annotation_geometry import review_annotations


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('manifest','archive','output'):parser.add_argument('--'+field,required=True)
    parser.add_argument('--selected',nargs='*',default=[])
    parser.add_argument('--allow-unverified-patients',action='store_true')
    args=parser.parse_args()
    print(json.dumps(review_annotations(args.manifest,args.archive,args.output,
        selected=args.selected,allow_unverified=args.allow_unverified_patients),indent=2))
