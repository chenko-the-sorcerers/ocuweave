"""Structural image-note audit without logging notes or unpickling NPZ objects."""
from collections import Counter, defaultdict
import csv
import hashlib
from pathlib import Path, PurePosixPath
import zipfile
from ocuweave.utils.provenance import sha256, write_json


def audit_fairvlmed(root, output):
    root, output = Path(root), Path(output)
    if output.exists():raise FileExistsError('Preserve prior audit; choose a new output')
    table = root/'ReadMe/data_summary.csv'
    archive = root/'Dataset/dataset.zip'
    with table.open(newline='',encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream)
        required={'filename','use','note','glaucoma'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Unexpected clinical-table schema; explicit review required')
        rows=list(reader)
    filenames=[r['filename'] for r in rows]
    if len(filenames)!=len(set(filenames)):raise ValueError('Repeated clinical-table filename')
    with zipfile.ZipFile(archive) as source:
        members=[n for n in source.namelist() if n.endswith('.npz') and '__MACOSX' not in n]
        if any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts for n in members):
            raise ValueError('Unsafe archive member path')
        index={PurePosixPath(n).name:n for n in members}
        if len(index)!=len(members):raise ValueError('Ambiguous NPZ filename pairing')
    splits=Counter(r['use'] for r in rows)
    if not set(splits).issubset({'training','validation','test','testing'}):
        raise ValueError('Unexpected official split labels')
    notes=defaultdict(set)
    missing_notes=0;keyword_flags=0;matched=0
    for row in rows:
        note=row['note'].strip()
        missing_notes += not note
        keyword_flags += 'glaucoma' in note.lower()
        matched += row['filename'] in index
        if note:notes[hashlib.sha256(note.encode()).hexdigest()].add(row['use'])
    result={
        'dataset':'FairVLMed','source':'https://huggingface.co/datasets/harvardairobotics/FairVLMed',
        'license':'CC-BY-NC-ND-4.0','csv_sha256':sha256(table),'archive_sha256':sha256(archive),
        'clinical_rows':len(rows),'npz_members':len(members),'paired_filenames':matched,
        'unmatched_rows':len(rows)-matched,'unreferenced_npz':len(set(index)-set(filenames)),
        'official_split_counts':dict(splits),'glaucoma_labels':dict(Counter(r['glaucoma'] for r in rows)),
        'missing_original_notes':missing_notes,'notes_containing_glaucoma_keyword':keyword_flags,
        'exact_nonempty_notes_across_official_splits':sum(len(v)>1 for v in notes.values()),
        'columns':list(rows[0]) if rows else [],'original_notes_preserved':True,
        'raw_notes_logged':False,'image_decoding_performed':False,'patient_linkage_verified':False,
        'limitations':[
            'Structural filename pairing only; no image or annotation cleaning is claimed.',
            'Diagnosis words can leak targets; keyword count is a screening flag, not semantic leakage adjudication.',
            'Repeated notes may be templates and do not establish repeated patients.',
            'Official splits are preserved. NPZ object arrays are not unpickled by this audit.',
            'Separate glaucoma cohort, never paired with OTFID records.']}
    output.mkdir(parents=True)
    write_json(output/'audit.json',result)
    return result
