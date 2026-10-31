"""Apply an OcuWeave four-channel IDRiD checkpoint to a fundus photograph."""
import argparse
from pathlib import Path
from ocuweave.inference.multilabel import save_lesion_prediction

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','image','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--save-probabilities',action='store_true')
    args=parser.parse_args()
    print(save_lesion_prediction(args.checkpoint,args.image,args.output,args.device,args.save_probabilities))
