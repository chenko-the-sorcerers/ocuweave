"""One isolated training process followed by development-only evaluation."""
import argparse
import json
from pathlib import Path
import torch
from ocuweave.training.engine import train
from ocuweave.evaluation.original_grid import evaluate_original
from ocuweave.utils.provenance import sha256, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('manifest','archive','output','config'):parser.add_argument('--'+field,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--allow-unverified-patients',action='store_true')
    args=parser.parse_args()
    if args.device=='cpu':torch.set_num_threads(2)
    config=json.loads(Path(args.config).read_text());root=Path(args.output)
    checkpoint=train(args.manifest,args.archive,root,config,
        allow_unverified=args.allow_unverified_patients,device=args.device)
    evaluate_original(args.manifest,args.archive,checkpoint,root/'validation-original.json',
        allow_unverified=args.allow_unverified_patients,device=args.device)
    files=('best.pt','checkpoint.json','provenance.json','history.json','validation-original.json')
    write_json(root/'complete.json',{'artifacts':{f:sha256(root/f) for f in files},'test_evaluated':False})


if __name__=='__main__':main()
