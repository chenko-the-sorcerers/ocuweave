"""Register, smoke-check or execute the locked nine-run exploratory protocol."""
import argparse
import json
from ocuweave.training.exploratory import lock,smoke,execute

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('register','smoke','execute'))
    for k in ('manifest','archive','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--smoke-report')
    a=p.parse_args()
    if a.action=='register':r=lock(a.manifest,a.archive,a.output)
    elif a.action=='smoke':r=smoke(a.manifest,a.archive,a.output,device=a.device)
    else:
        if not a.smoke_report:p.error('--smoke-report is required for execution')
        r=execute(a.manifest,a.archive,a.output,a.smoke_report)
    print(json.dumps(r,indent=2))
