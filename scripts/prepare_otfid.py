import argparse
import json
from ocuweave.data.adapters.otfid import audit_otfid

if __name__ == '__main__':
    p=argparse.ArgumentParser(description='Audit OTFID and freeze duplicate-aware splits.')
    p.add_argument('--archive',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--patient-map')
    p.add_argument('--image-column');p.add_argument('--label-column')
    p.add_argument('--allow-unverified-patients',action='store_true',help='Explicit exploratory image split; NOT patient-independent.')
    a=p.parse_args()
    print(json.dumps(audit_otfid(a.archive,a.output,seed=a.seed,patient_map=a.patient_map,
        image_column=a.image_column,label_column=a.label_column,allow_unverified_patients=a.allow_unverified_patients),indent=2))
