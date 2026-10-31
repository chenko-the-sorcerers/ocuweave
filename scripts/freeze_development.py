"""Freeze a new development manifest without altering historical test rows."""
import argparse
import json
from ocuweave.data.splitting.development import freeze

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','duplicates','audit-summary','quality-csv','output'):parser.add_argument('--'+name,required=True)
    a=parser.parse_args()
    result=freeze(a.manifest,a.duplicates,a.audit_summary,a.quality_csv,a.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('changes','provisional_groups')},indent=2))
