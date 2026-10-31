"""FIVES release preparation with immutable official partitions and similarity groups."""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

from ocuweave.data.auditing.similarity import group_similar
from ocuweave.data.preprocessing.geometry import letterbox
from ocuweave.utils.provenance import sha256, write_json


def read_pair(root, row):
    with Image.open(Path(root)/row['image']) as source:
        image = source.convert('RGB')
    with Image.open(Path(root)/row['mask']) as source:
        values = np.asarray(source.convert('L'))
    if values.shape != (image.height, image.width):
        raise ValueError('Image-mask geometry mismatch: '+row['sample_id'])
    if not set(np.unique(values)).issubset({0, 1, 255}):
        raise ValueError('Unexpected mask encoding: '+row['sample_id'])
    return image, values > 0


def prepare_fives(root, output, *, seed=42):
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous manifests')
    rows, thumbs = [], {}
    for image_path in sorted(root.rglob('*.png')):
        parts = image_path.relative_to(root).parts
        if '__MACOSX' in parts or image_path.parent.name.lower() != 'original':
            continue
        partitions = [p.lower() for p in parts if p.lower() in {'train', 'test'}]
        if len(partitions) != 1:
            raise ValueError('Unknown official partition')
        siblings = [p for p in image_path.parent.parent.iterdir()
                    if p.is_dir() and p.name.lower().replace(' ', '') == 'groundtruth']
        if len(siblings) != 1:
            raise ValueError('Expected one Ground truth directory')
        mask_path = siblings[0]/image_path.name
        if not mask_path.is_file():
            raise ValueError('Missing paired mask; not assumed negative')
        row = {'sample_id':partitions[0]+'-'+image_path.stem,
               'image':str(image_path.relative_to(root)), 'mask':str(mask_path.relative_to(root)),
               'official_split':partitions[0], 'disease_stratum':image_path.stem.split('_')[-1]}
        image, target = read_pair(root, row)
        pixels = np.asarray(image)
        small = np.asarray(image.convert('L').resize((9, 8), Image.Resampling.BILINEAR))
        row.update(width=image.width, height=image.height, vessel_pixels=int(target.sum()),
                   image_sha256=sha256(image_path), mask_sha256=sha256(mask_path),
                   pixel_sha256=hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest(),
                   dhash=f'{sum(int(b)<<i for i,b in enumerate((small[:,1:]>small[:,:-1]).flat)):016x}')
        thumbs[row['sample_id']] = np.asarray(image.resize((64,64), Image.Resampling.BILINEAR),dtype=np.float32)/255
        rows.append(row)
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Missing data or duplicate sample identifiers')
    candidates = group_similar(rows, thumbs)
    groups = defaultdict(list)
    for row in rows:
        groups[row['similarity_group']].append(row)
    # Never repair an official train/test overlap silently.
    for group in groups.values():
        if len({r['official_split'] for r in group}) != 1:
            raise ValueError('Cross-official-split similarity group requires adjudication')
    exact = defaultdict(set)
    for row in rows:
        exact[row['pixel_sha256']].add(row['official_split'])
    if any(len(s)>1 for s in exact.values()):
        raise ValueError('Exact duplicate crosses official partitions')
    strata = defaultdict(list)
    for group_id, group in groups.items():
        if group[0]['official_split']=='train':
            strata[group[0]['disease_stratum']].append(group_id)
    validation_groups = set()
    rng = random.Random(seed)
    for stratum in sorted(strata):
        ids = sorted(strata[stratum]); rng.shuffle(ids)
        if len(ids)<2:
            raise ValueError('Need at least two training groups per stratum')
        validation_groups.update(ids[:max(1,round(.2*len(ids)))])
    for row in rows:
        row['split'] = ('test' if row['official_split']=='test' else
                        'validation' if row['similarity_group'] in validation_groups else 'train')
    result = {'protocol':'fives-development-v1','seed':seed,'dataset':'FIVES',
              'source':'https://doi.org/10.6084/m9.figshare.19688169.v1','license':'CC-BY-4.0',
              'patient_identity_verified':False,'split_unit':'provisional_similarity_group',
              'records':rows,'counts':dict(Counter(r['split'] for r in rows)),
              'similarity_candidates':candidates,
              'policy':'Official test preserved; 20% of training groups per disease stratum used for validation; no quality exclusions.',
              'limitations':['Patient/eye identity unavailable; similarity is not identity verification.',
                             'Technical geometry checks do not validate clinical boundaries.']}
    write_json(output, result)
    return result


class FivesDataset(Dataset):
    def __init__(self, root, manifest, split, size=192, augment=False):
        self.root=Path(root); self.size=size; self.augment=augment; self.cache={}
        self.rows=[r for r in manifest['records'] if r['split']==split]
        if not self.rows: raise ValueError('Empty split')

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        if index not in self.cache:
            image, target=read_pair(self.root,self.rows[index])
            self.cache[index]=letterbox(image,target,self.size)
        image, target, valid=(a.copy() for a in self.cache[index])
        if self.augment and random.random()<.5:
            image=image[:,::-1].copy(); target=target[:,::-1].copy(); valid=valid[:,::-1].copy()
        return {'image':torch.from_numpy(image.transpose(2,0,1)).float()/255,
                'mask':torch.from_numpy(target[None]), 'valid':torch.from_numpy(valid[None]),
                'id':self.rows[index]['sample_id']}


def verify_fives(root, manifest):
    groups={}; pixels={}
    for row in manifest['records']:
        for field, mapping in [('similarity_group',groups),('pixel_sha256',pixels)]:
            value=row[field]
            if value in mapping and mapping[value]!=row['split']:
                raise ValueError('Cross-split group or duplicate')
            mapping[value]=row['split']
        for field in ('image','mask'):
            relative=Path(row[field])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe data path')
            if sha256(Path(root)/relative)!=row[field+'_sha256']:
                raise ValueError('Data changed since manifest freeze')
