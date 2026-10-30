"""Generate reproducible QC or prediction panels and learning curves."""
import argparse
from pathlib import Path
from ocuweave.evaluation.figures import sample_panel, learning_curves

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manifest',required=True)
parser.add_argument('--archive',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--checkpoint')
parser.add_argument('--split',choices=['train','validation','test'],default='train')
parser.add_argument('--query-run')
parser.add_argument('--fcn-run')
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True,exist_ok=True)
sample_panel(args.manifest,args.archive,output/f'{args.split}_panel.png',args.checkpoint,args.split)
if args.query_run and args.fcn_run:
    learning_curves({'query':args.query_run,'fcn':args.fcn_run},output/'learning_curves.png')
