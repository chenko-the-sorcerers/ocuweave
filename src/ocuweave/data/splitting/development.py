"""Conservative development-only regrouping; historical test rows stay byte-equivalent."""
import copy
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from ocuweave.utils.provenance import sha256, write_json


def regroup(data, pairs):
    result=copy.deepcopy(data)
    rows=result['records'];lookup={r['sample_id']:r for r in rows}
    if len(lookup)!=len(rows):raise ValueError('Duplicate sample identifiers')
    if data['patient_identity_verified']:raise ValueError('This protocol is for unverified exploratory identities')
    if pairs.get('truncated'):raise ValueError('Truncated candidate screen requires review')
    parent={key:key for key in lookup}
    def find(key):
        while parent[key]!=key:
            parent[key]=parent[parent[key]];key=parent[key]
        return key
    def union(a,b):parent[find(a)]=find(b)
    seen={}
    for row in rows:
        for field in ('split_group','pixel_sha256','similarity_group'):
            if row.get(field):
                key=(field,row[field])
                if key in seen:union(row['sample_id'],seen[key])
                seen[key]=row['sample_id']
        for pid in set(row.get('related_patient_ids',[])+([row['patient_id']] if row.get('patient_id') else [])):
            key=('patient',pid)
            if key in seen:union(row['sample_id'],seen[key])
            seen[key]=row['sample_id']
    candidate_ids=set()
    for pair in pairs['pairs']:
        a,b=pair['a'],pair['b']
        if a not in lookup or b not in lookup or a==b:raise ValueError('Invalid candidate membership')
        if lookup[a]['split']!=pair['a_split'] or lookup[b]['split']!=pair['b_split']:
            raise ValueError('Candidate split differs from source manifest')
        if 'test' in (lookup[a]['split'],lookup[b]['split']):
            raise ValueError('Test-involving candidate requires a separate protocol')
        union(a,b);candidate_ids.update((a,b))
    groups=defaultdict(list)
    for row in rows:groups[find(row['sample_id'])].append(row)
    changes=[];decisions=[]
    for members in groups.values():
        ids=sorted(r['sample_id'] for r in members)
        splits={r['split'] for r in members}
        if 'test' in splits:
            if len(splits)>1:raise ValueError('Development/test group bridge; stop')
            continue
        if not candidate_ids.intersection(ids):continue
        # Conservative deterministic rule, independent of image quality or model scores.
        target='train' if 'train' in splits else 'validation'
        group='provisional-'+hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
        for row in members:
            if row['split']!=target:changes.append({'sample_id':row['sample_id'],'from':row['split'],'to':target})
            row.update(split=target,split_group=group,similarity_group=group,
                       provisional_image_family=group,patient_identity_verified=False)
        decisions.append({'group':group,'members':ids,'destination':target,
                          'status':'provisional image family; not patient identity'})
    if not any(r['split']=='validation' for r in rows):raise ValueError('Empty validation after grouping')
    assert [r for r in rows if r['split']=='test']==[r for r in data['records'] if r['split']=='test']
    result.update(split_unit='provisional_image_family_exploratory',patient_identity_verified=False)
    # Update known aggregate metadata; preserve historical source in the immutable parent file.
    result['split_counts']=dict(Counter(r['split'] for r in rows))
    return result,{'protocol':'otfid-development-grouping-v1','rule':'Move an entire candidate-connected development component to training if it contains a training member; otherwise retain validation.',
        'counts_before':dict(Counter(r['split'] for r in data['records'])),
        'counts_after':result['split_counts'],'changes':changes,'provisional_groups':decisions,
        'test_rows_unchanged':True,'quality_exclusions':0,'patient_identity_verified':False,
        'clinical_annotation_review_complete':False,'emr_used':False}


def freeze(manifest,duplicates,audit_summary,quality_csv,output):
    output=Path(output)
    if output.exists():raise FileExistsError('Preserve every frozen development manifest')
    data=json.loads(Path(manifest).read_text());audit=json.loads(Path(audit_summary).read_text())
    if audit['otfid']['manifest_sha256']!=sha256(manifest):raise ValueError('Audit/source manifest mismatch')
    if audit['otfid']['archive_sha256']!=data['archive_sha256']:raise ValueError('Audit/archive mismatch')
    if Path(duplicates).resolve().parent.parent!=Path(audit_summary).resolve().parent:
        raise ValueError('Candidates must belong to the supplied audit')
    pairs=json.loads(Path(duplicates).read_text())
    if len(pairs['pairs'])!=audit['otfid']['near_duplicate_candidates']:raise ValueError('Candidate count mismatch')
    with Path(quality_csv).open(newline='') as stream:quality=list(csv.DictReader(stream))
    dev={r['sample_id'] for r in data['records'] if r['split']!='test'}
    if len(quality)!=len(dev) or {r['sample_id'] for r in quality}!=dev:raise ValueError('Quality coverage mismatch')
    revised,decision=regroup(data,pairs)
    flags={r['sample_id']:int(r['flag_count']) for r in quality}
    for row in revised['records']:
        if row['split']!='test':row['historical_quality_flag_count']=flags[row['sample_id']]
    decision['flagged_development_retained']=sum(v>0 for v in flags.values())
    decision['source_hashes']={name:sha256(path) for name,path in
        [('manifest',manifest),('duplicates',duplicates),('audit_summary',audit_summary),('quality_csv',quality_csv)]}
    revised['development_revision']={'protocol':decision['protocol'],'parent_manifest_sha256':sha256(manifest),
        'candidate_file_sha256':sha256(duplicates),'quality_policy':'retain all flags pending review'}
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'manifest.json',revised)
    decision['manifest_sha256']=sha256(output/'manifest.json')
    write_json(output/'decision.json',decision)
    return decision
