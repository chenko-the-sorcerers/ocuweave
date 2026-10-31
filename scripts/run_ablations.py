"""Register or execute the three-variant, three-seed development protocol."""
import argparse
from ocuweave.training.ablation import run_suite


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('manifest','archive','output'):parser.add_argument('--'+field,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    result=run_suite(args.manifest,args.archive,args.output,device=args.device,execute=args.execute)
    print({r['run']:r['status'] for r in result['runs']})
