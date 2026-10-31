import argparse
from ocuweave.data.adapters.idrid import prepare_idrid
from ocuweave.training.lesions import run_idrid

if __name__=='__main__':
    p=argparse.ArgumentParser(description='IDRiD four-channel lesion experiments')
    p.add_argument('action',choices=['prepare','pilot']);p.add_argument('--root',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--output');p.add_argument('--steps',type=int,default=100)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    if a.action=='prepare':print(prepare_idrid(a.root,a.manifest)['counts'])
    else:
        if not a.output:p.error('--output required')
        run_idrid(a.root,a.manifest,a.output,a.steps,a.device)
