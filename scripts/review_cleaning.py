"""Read-only quality/context/linkage screening. No deletion, relabelling or split edits."""
import csv
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile
import argparse
import subprocess
import numpy as np
from PIL import Image
from ocuweave.data.adapters.otfid import read_image
from ocuweave.data.auditing.fairvlmed_images import safe_array,MAX_NPZ_BYTES
from ocuweave.data.auditing.quality_screen import quality_metrics,fit_thresholds,quality_flags,descriptor,near_pairs
from ocuweave.data.auditing.note_context import review_context
from ocuweave.training.engine import checked_manifest
from ocuweave.utils.provenance import sha256,write_json


def write_csv(path, rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def quality_report(rows,output):
    thresholds=fit_thresholds(rows)
    for row in rows:
        flags=quality_flags(row,thresholds)
        row.update(flags,flag_count=sum(flags.values()),clinical_gradability='unreviewed',reviewer='',decision='pending')
    counts={key:sum(r[key] for r in rows) for key in quality_flags(rows[0],thresholds)}
    write_csv(output/'image-quality-review.csv',rows)
    result={'samples':len(rows),'threshold_fit_splits':['train','training'],'thresholds':thresholds,
        'flag_counts':counts,'images_with_any_flag':sum(r['flag_count']>0 for r in rows),
        'automatically_removed':0,'clinical_quality_verified':False,
        'limitations':['Quantile outliers are cohort-relative review priorities, not bad-image diagnoses.',
            'Non-dark support and edge contact are not retinal-field segmentation; SLO acquisition borders differ from color fundus.',
            'Clipping/asymmetry may reflect normal anatomy or disease, not artifacts. Clinical review required.']}
    write_json(output/'quality-summary.json',result)
    return result


def linkage_evidence(columns,rows,dataset):
    candidates=[c for c in columns if re.search(r'patient|subject|mrn|encounter|visit|date',c,re.I)]
    known=sum(bool(r.get('patient_id')) for r in rows) if dataset=='OTFID' else None
    return {'dataset':dataset,'exported_columns':columns,'candidate_identity_or_time_columns':candidates,
        'manifest_rows_with_patient_id':known,'verified_mapping_supplied':False,
        'patient_independence_verified':False,'patient_or_visit_ids_inferred_from_filename':False,
        'source_reported_unique_patients':10000 if dataset=='FairVLMed' else None,
        'decision':'Keep official/frozen splits; no independent patient-level guarantee. Obtain a documented de-identified patient/eye/visit mapping from the dataset custodian.',
        'cross_dataset_patient_join_performed':False}


def plot_quality(samples,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for number,(row,image) in enumerate(samples):
        fig,ax=plt.subplots(figsize=(6,7));fig.subplots_adjust(bottom=.23,top=.92)
        ax.imshow(image,cmap='gray' if image.mode=='L' else None);ax.axis('off')
        flags=[k for k,v in row.items() if k.endswith('_proxy') and v]
        ax.set_title(row['sample_id']+' · technical review candidate',fontsize=10)
        fig.text(.02,.01, '\n'.join(flags) or 'Unflagged control; not clinically cleared',fontsize=8)
        fig.savefig(output/f'quality-example-{number:02d}.png',dpi=120);plt.close(fig)


def otfid(archive,manifest,output):
    data=checked_manifest(manifest,archive,True);all_rows=data['records'];quality=[];descriptors=[]
    with zipfile.ZipFile(archive) as source:
        csv_name=next(n for n in source.namelist() if PurePosixPath(n).name=='dataset_labels.csv' and '__MACOSX' not in n)
        header=source.read(csv_name).decode('utf-8-sig').splitlines()[0]
        columns=next(csv.reader([header],delimiter=';' if header.count(';')>header.count(',') else ','))
        for row in all_rows:
            image=read_image(source,row['image'])
            descriptors.append(descriptor(image))
            if row['split'] in {'train','validation'}:
                quality.append({'sample_id':row['sample_id'],'split':row['split'],'source':row.get('source'),**quality_metrics(image)})
    output.mkdir(parents=True)
    q=quality_report(quality,output)
    duplicates=near_pairs(all_rows,descriptors)
    write_json(output/'near-duplicates.json',duplicates)
    write_json(output/'patient-linkage.json',linkage_evidence(columns,all_rows,'OTFID'))
    lookup={r['sample_id']:r for r in quality};annotation=[]
    for row in all_rows:
        if row['split'] not in {'train','validation'}:continue
        annotation.append({'sample_id':row['sample_id'],'split':row['split'],'source':row.get('source'),
            'label':row['label'],'mask_files':len(row['masks']),'healthy_zero_mask':row['healthy_zero_mask'],
            'lesion_fraction':row['lesion_pixels']/(row['width']*row['height']),
            'small_lesion_review':0<row['lesion_pixels']/(row['width']*row['height'])<=.01,
            'quality_flag_count':lookup[row['sample_id']]['flag_count'],
            'lesion_location_correct':'unreviewed','boundary_complete':'unreviewed',
            'missing_lesion_regions':'unreviewed','activity_union_appropriate':'unreviewed',
            'reviewer':'','review_date':'','decision':'pending','rationale':''})
    write_csv(output/'annotation-clinical-review.csv',annotation)
    ranked=sorted(quality,key=lambda r:(-r['flag_count'],r['sample_id']))
    selected=ranked[:4]+sorted(quality,key=lambda r:(r['flag_count'],r['sample_id']))[:2]
    metadata={r['sample_id']:r for r in all_rows}
    with zipfile.ZipFile(archive) as source:
        plot_quality([(r,read_image(source,metadata[r['sample_id']]['image'])) for r in selected],output)
    return {'quality':q,'fingerprinted_images':len(all_rows),'near_duplicate_candidates':len(duplicates['pairs']),
        'cross_split_candidates':sum(p['cross_split'] for p in duplicates['pairs']),
        'duplicate_scan_truncated':duplicates['truncated'],'annotation_review_rows':len(annotation),
        'manifest_sha256':sha256(manifest),'archive_sha256':data['archive_sha256']}


def fairvlmed(root,output):
    table=root/'ReadMe/data_summary.csv';archive=root/'Dataset/dataset.zip'
    expected={r['path']:r['sha256'] for r in json.loads((root/'download_manifest.json').read_text())['files']}
    for relative,path in [('ReadMe/data_summary.csv',table),('Dataset/dataset.zip',archive)]:
        if sha256(path)!=expected[relative]:raise ValueError('FairVLMed source receipt mismatch')
    with table.open(newline='',encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream);columns=reader.fieldnames;rows=list(reader)
    if not rows or not {'filename','use','note','glaucoma'}.issubset(columns):raise ValueError('Unexpected clinical table schema')
    if any(r['use'] not in {'training','validation','test'} for r in rows):raise ValueError('Unexpected split')
    if len({r['filename'] for r in rows})!=len(rows):raise ValueError('Ambiguous filename pairing')
    quality=[];descriptors=[];identities=[]
    with zipfile.ZipFile(archive) as source:
        names=[n for n in source.namelist() if n.endswith('.npz') and '__MACOSX' not in n]
        index={PurePosixPath(n).name:n for n in names}
        if len(index)!=len(names):raise ValueError('Ambiguous NPZ members')
        for number,row in enumerate(rows,1):
            member=index[row['filename']]
            if source.getinfo(member).file_size>MAX_NPZ_BYTES:raise ValueError('Oversized numeric image member')
            with zipfile.ZipFile(io.BytesIO(source.read(member))) as inner:
                values=safe_array(inner,'slo_fundus.npy')
            if values.dtype!=np.uint8 or values.ndim!=2:raise ValueError('Review expects verified uint8 grayscale release')
            image=Image.fromarray(values)
            identities.append({'sample_id':row['filename'],'split':row['use']})
            descriptors.append(descriptor(image))
            if row['use'] in {'training','validation'}:
                quality.append({'sample_id':row['filename'],'split':row['use'],**quality_metrics(image)})
            if number%1000==0:print('FairVLMed numeric images processed:',number,flush=True)
    output.mkdir(parents=True)
    q=quality_report(quality,output)
    duplicates=near_pairs(identities,descriptors)
    write_json(output/'near-duplicates.json',duplicates)
    notes=review_context(rows)
    write_json(output/'note-context-review.json',notes)
    write_csv(output/'emr-clinical-review.csv',[{'filename':r['filename'],'split':r['split'],
        'priority_flagged':r['requires_review'],'diagnosis_mentions':len(r['mentions']),
        'assertion':'unreviewed','experiencer':'unreviewed','laterality':'unreviewed',
        'prediction_time':'unknown','note_before_prediction':'unverified','treatment_or_endpoint_leakage':'unreviewed',
        'reviewer':'','review_date':'','decision':'pending','rationale_without_note_text':''} for r in notes['review_queue']])
    write_json(output/'patient-linkage.json',linkage_evidence(columns,rows,'FairVLMed'))
    selected=sorted(quality,key=lambda r:(-r['flag_count'],r['sample_id']))[:4]+sorted(quality,key=lambda r:(r['flag_count'],r['sample_id']))[:2]
    with zipfile.ZipFile(archive) as source:
        samples=[]
        for row in selected:
            with zipfile.ZipFile(io.BytesIO(source.read(index[row['sample_id']]))) as inner:
                image=Image.fromarray(safe_array(inner,'slo_fundus.npy'))
            samples.append((row,image))
        plot_quality(samples,output)
    return {'quality':q,'fingerprinted_images':len(rows),'near_duplicate_candidates':len(duplicates['pairs']),
        'cross_split_candidates':sum(p['cross_split'] for p in duplicates['pairs']),
        'duplicate_scan_truncated':duplicates['truncated'],'note_context_counts':notes['counts'],
        'csv_sha256':sha256(table),'archive_sha256':sha256(archive)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','manifest','fairvlmed-root','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError('Preserve existing cleaning audits')
    output.mkdir(parents=True)
    o=otfid(args.archive,args.manifest,output/'otfid')
    print('OTFID completed:',json.dumps(o),flush=True)
    f=fairvlmed(Path(args.fairvlmed_root),output/'fairvlmed')
    report={'protocol':'cleaning-review-v1','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'otfid':o,'fairvlmed':f,'automatic_exclusions':0,'annotations_changed':0,'splits_changed':False,
        'test_images_used_for':'Numerical near-duplicate fingerprints only; no test previews, quality tuning or model predictions',
        'test_note_content_analyzed':False,'patient_linkage_verified':False,'clinical_annotation_review_complete':False,
        'emr_training_cleared':False,'training_performed':False}
    write_json(output/'summary.json',report)
    (output/'README.md').write_text('# Cleaning review\n\nTechnical flags and reviewer queues only. No original record has been changed.\n\n```json\n'+json.dumps(report,indent=2)+'\n```\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
