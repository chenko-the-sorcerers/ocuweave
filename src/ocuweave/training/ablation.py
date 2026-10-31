"""Predeclared paired-seed runs; partial runs never masquerade as results."""
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
from ocuweave.training.engine import validate_config
from ocuweave.utils.provenance import sha256, write_json


def configurations():
    base=dict(width=24,dim=48,queries=8,batch_size=8,epochs=20,learning_rate=.001)
    return {f'{name}-seed{seed}':dict(base,architecture=arch,size=size,seed=seed)
        for seed in (42,123,2026)
        for name,arch,size in (('query192','query',192),('query384','query',384),
                               ('detail384','query_detail',384))}


def summarize(root, configs):
    """Read only complete runs; retain status for every predeclared run."""
    root=Path(root);rows=[]
    contract=json.loads((root/'protocol.json').read_text()) if (root/'protocol.json').exists() else None
    for name,config in configs.items():
        folder=root/name
        row={'run':name,**config,'status':'pending'}
        row.update({key:None for key in (
            'source_commit','manifest_sha256','archive_sha256','checkpoint_sha256',
            'positive_dice','positive_iou','macro_iou','small_lesion_dice','small_lesion_count',
            'healthy_any_pixel_fp','healthy_mean_fp_area','elapsed_seconds','evaluation_seconds',
            'optimizer_steps','parameters','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes',
            'peak_process_rss_bytes','device','device_name')})
        if contract:
            row.update({key:contract[key] for key in ('source_commit','manifest_sha256','archive_sha256')})
        if folder.exists():row['status']='incomplete'
        if (folder/'complete.json').exists():
            receipt=json.loads((folder/'complete.json').read_text())
            for filename,digest in receipt['artifacts'].items():
                if sha256(folder/filename)!=digest:raise ValueError('Completed artifact changed')
            report=json.loads((folder/'validation-original.json').read_text())
            timing=json.loads((folder/'checkpoint.json').read_text())
            provenance=json.loads((folder/'provenance.json').read_text())
            if provenance['config']!=config:raise ValueError('Run configuration mismatch')
            if contract and any(provenance[k]!=contract[k] for k in ('source_commit','manifest_sha256','archive_sha256')):
                raise ValueError('Run provenance differs from registered suite')
            if timing['optimizer_steps_completed']!=provenance['optimizer_steps_planned']:
                raise ValueError('Incomplete optimizer budget')
            small=report['by_original_area'].get('small_le_1pct',{})
            row.update(status='completed',
                manifest_sha256=provenance['manifest_sha256'],archive_sha256=provenance['archive_sha256'],
                checkpoint_sha256=report['checkpoint_sha256'],positive_dice=report['metrics']['positive_macro_dice'],
                small_lesion_dice=small.get('positive_macro_dice'),
                positive_iou=report['metrics']['positive_macro_iou'],macro_iou=report['metrics']['macro_iou'],
                small_lesion_count=small.get('positive_samples',0),
                evaluation_seconds=report['elapsed_seconds_including_decode'],
                healthy_any_pixel_fp=report['metrics']['empty_image_false_positive_rate'],
                healthy_mean_fp_area=report['mean_healthy_false_positive_area_fraction'],
                elapsed_seconds=timing['elapsed_seconds'],optimizer_steps=timing['optimizer_steps_completed'],
                peak_cuda_allocated_bytes=timing['peak_cuda_allocated_bytes'],
                peak_cuda_reserved_bytes=timing['peak_cuda_reserved_bytes'],
                peak_process_rss_bytes=timing['peak_process_rss_bytes'],
                parameters=provenance['parameters'],source_commit=provenance['source_commit'],
                device=provenance['device'],device_name=provenance['device_name'])
        rows.append(row)
    paired=[]
    for left,right,label in [('query192','query384','resolution'),('query384','detail384','local_detail')]:
        pairs=[]
        for seed in sorted({c['seed'] for c in configs.values()}):
            lookup={r['run']:r for r in rows if r['status']=='completed'}
            a,b=lookup.get(f'{left}-seed{seed}'),lookup.get(f'{right}-seed{seed}')
            if a is not None and b is not None:
                pairs.append({'seed':seed,'positive_dice_delta':b['positive_dice']-a['positive_dice']})
        differences=[r['positive_dice_delta'] for r in pairs]
        paired.append({'contrast':label,'pairs':pairs,'n_paired_seeds':len(pairs),
            'mean_delta':statistics.mean(differences) if differences else None,
            'sample_sd_delta':statistics.stdev(differences) if len(differences)>1 else None,
            'interpretation':'Descriptive seed variability, not patient-level uncertainty or a significance test'})
    groups={}
    for variant in sorted({name.split('-seed')[0] for name in configs}):
        selected=[r for r in rows if r['status']=='completed' and r['run'].split('-seed')[0]==variant]
        groups[variant]={'completed_seeds':len(selected),'metrics':{}}
        for metric in ('positive_dice','positive_iou','macro_iou','small_lesion_dice','healthy_any_pixel_fp','healthy_mean_fp_area','elapsed_seconds','peak_cuda_allocated_bytes'):
            values=[r[metric] for r in selected if r.get(metric) is not None]
            groups[variant]['metrics'][metric]={'n':len(values),'mean':statistics.mean(values) if values else None,
                'sample_sd':statistics.stdev(values) if len(values)>1 else None}
    result={'runs':rows,'variant_summaries':groups,'paired_comparisons':paired,'test_evaluated':False}
    write_json(root/'summary.json',result)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (root/'runs.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    lines=['# Development experiment register','',
        'Original-image-grid validation at threshold 0.5. Pending runs have no measured result.',
        '', '| Run | Status | Positive Dice | Small-lesion Dice | Healthy mean FP area |',
        '|---|---|---:|---:|---:|']
    for r in rows:
        values=[f"{r[k]:.4f}" if r.get(k) is not None else '—' for k in ('positive_dice','small_lesion_dice','healthy_mean_fp_area')]
        lines.append(f"| {r['run']} | {r['status']} | "+' | '.join(values)+' |')
    lines+=['','Full configuration, timing, device, steps, parameter and memory counts: `runs.csv`.',
        'Paired seed differences: `summary.json`. Do not compare these values directly with input-grid pilot scores.']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    return result


def run_suite(manifest, archive, output, *, device='cuda', execute=False, configs=None):
    configs=configs or configurations()
    for config in configs.values():validate_config(config)
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[3]
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():
        raise ValueError('Commit source before creating a reproducible suite')
    contract={'protocol':'otfid-resolution-detail-v1','source_commit':revision,
              'manifest_sha256':sha256(manifest),'archive_sha256':sha256(archive),
              'configs':configs,'checkpoint_selection':'best validation positive Dice on input grid',
              'comparison_grid':'original image','test_access':'none'}
    contract_path=root/'protocol.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text())!=contract:raise ValueError('Suite contract changed; use a new folder')
    else:write_json(contract_path,contract)
    summary=summarize(root,configs)
    if not execute:return summary
    for name,config in configs.items():
        target=root/name
        if (target/'complete.json').exists():continue
        if target.exists():raise FileExistsError(f'Incomplete run {name}; preserve it and choose a new suite')
        config_path=root/f'{name}.json';write_json(config_path,config)
        args=[sys.executable,str(repo/'scripts/run_development_experiment.py'),
              '--manifest',str(manifest),'--archive',str(archive),'--output',str(target),
              '--config',str(config_path),'--device',device,'--allow-unverified-patients']
        with (root/f'{name}.log').open('x') as log:
            status=subprocess.run(args,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
        summarize(root,configs)
        if status.returncode:raise RuntimeError(f'Run failed; inspect {name}.log. No automatic retry or altered batch size.')
    return summarize(root,configs)
