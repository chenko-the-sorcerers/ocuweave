"""Snapshot public documentation and inventory private research artifacts in Drive."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
from ocuweave.utils.provenance import sha256, write_json


def measured_reports(project):
    """Retain every aggregate field; link high-cardinality rows to source JSON."""
    patterns=(
        '02 — Audits/otfid-v3-binary-v2-seed42-v0.2.0/audit.json',
        '02 — Audits/quality-review-v0.3.0/fairvlmed/summary.json',
        '02 — Audits/quality-review-v0.3.0/fairvlmed/note_review.json',
        '02 — Audits/quality-review-v0.3.0/otfid-*/diagnostics.json',
        '02 — Audits/development-v0.4.0/annotation-geometry/summary.json',
        '02 — Audits/cleaning-review-v0.5.0/*/protected-*.json',
        '02 — Audits/cleaning-review-v0.5.0/*/*/*.json',
        '02 — Audits/cleaning-review-v0.5.0/*/*/*/*.json',
        '04 — Runs/v0.2.0/pilot-seed42-20epochs/*-test.json',
        '04 — Runs/v0.2.0/pilot-seed42-20epochs/*/provenance.json',
        '04 — Runs/v0.2.0/pilot-seed42-20epochs/*/checkpoint.json',
        '04 — Runs/v0.2.0/pilot-seed42-20epochs/*/history.json',
        '04 — Runs/v0.4.0/resolution-detail-seeds/summary.json',
        '02 — Audits/exploratory-v0.6.0/*/*.json',
        '02 — Audits/exploratory-v0.6.0/*/*/*.json',
        '04 — Runs/v0.6.0/*/resolution-detail-seeds/*.json',
        '04 — Runs/v0.6.0/*/resolution-detail-seeds/*/*.json')
    reports=[]
    for pattern in patterns:
        for path in sorted(project.glob(pattern)):
            data=json.loads(path.read_text())
            if isinstance(data,dict):
                data={k:({'rows':len(v),'details':'See source artifact and its recorded SHA-256'}
                         if k in {'per_image','records','pairs','review_queue','normalized_duplicate_groups','exclusions','warnings'} else v)
                      for k,v in data.items()}
            reports.append({'source':str(path.relative_to(project)),'sha256':sha256(path),'result':data})
    return reports


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
    reports=measured_reports(project)
    write_json(output/'measured-results.json',reports)
    content=['# Consolidated measured results','',
        'Machine-generated from the saved source reports below. Historical input-grid and new original-grid metrics are different endpoints.',
        'Per-image rows and review queues remain in their source files, indexed by SHA-256. Pending runs have no measured performance.','']
    for report in reports:
        content += ['## '+report['source'],'','SHA-256: `'+report['sha256']+'`','',
                    '```json',json.dumps(report['result'],indent=2,ensure_ascii=False),'```','']
    (output/'measured-results.md').write_text('\n'.join(content))
    return {'snapshot':str(output),'artifacts_indexed':len(items)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(snapshot(args.project,args.output))
