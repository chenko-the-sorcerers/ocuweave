"""Prepare and run reproducible FIVES experiments from an extracted official release."""
import argparse
from ocuweave.data.adapters.fives import prepare_fives
from ocuweave.training.vessels import run_pilot

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','pilot'])
    parser.add_argument('--root',required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output')
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    if args.action=='prepare':
        result=prepare_fives(args.root,args.manifest);print(result['counts'])
    else:
        if not args.output:parser.error('--output required for pilot')
        run_pilot(args.root,args.manifest,args.output,steps=args.steps,device=args.device)
