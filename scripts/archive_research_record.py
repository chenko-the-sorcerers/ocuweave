"""Snapshot public documentation and inventory private research artifacts in Drive."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
from ocuweave.utils.provenance import sha256, write_json


def snapshot(project, output):
    project,output=Path(project),Path(output)
    if output.exists():raise FileExistsError('Choose a new documentation snapshot')
    repo=Path(__file__).resolve().parents[1]
    output.mkdir(parents=True)
    shutil.copytree(repo/'docs/results',output/'results')
    shutil.copytree(repo/'docs/protocols',output/'protocols')
    for filename in ('CHANGELOG.md','CITATION.cff','pyproject.toml'):
        shutil.copy2(repo/filename,output/filename)
    items=[]
    for folder in ('02 — Audits','03 — Notebooks','04 — Runs'):
        for path in sorted((project/folder).rglob('*')):
            if path.is_file() and path.suffix.lower() in {'.json','.csv','.md','.png','.log','.ipynb','.pt'}:
                items.append({'path':str(path.relative_to(project)),'bytes':path.stat().st_size,'sha256':sha256(path)})
    write_json(output/'inventory.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        'created_utc':datetime.now(timezone.utc).isoformat(),'artifacts':items,
        'scope':'Private audit/run/notebook artifacts; raw dataset hashes are in acquisition receipts. No clinical notes exported.'})
    return {'snapshot':str(output),'artifacts_indexed':len(items)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(snapshot(args.project,args.output))
