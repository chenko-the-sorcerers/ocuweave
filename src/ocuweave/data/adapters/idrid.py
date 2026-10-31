"""Four-channel IDRiD segmentation; an absent annotation is unknown, never background."""
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from ocuweave.data.auditing.similarity import group_similar
from ocuweave.data.preprocessing.geometry import letterbox
from ocuweave.utils.provenance import sha256,write_json

CLASSES=('MA','HE','EX','SE')


def read_idrid(root,row):
    with Image.open(Path(root)/row['image']) as f:image=f.convert('RGB')
    targets=[];available=[]
    for name in CLASSES:
        path=row['masks'][name];available.append(path is not None)
        if path is None:targets.append(np.zeros((image.height,image.width),dtype=bool));continue
        with Image.open(Path(root)/path) as f:mask=np.asarray(f.convert('L'))
        if mask.shape!=(image.height,image.width):raise ValueError('Mask geometry mismatch')
        if not set(np.unique(mask)).issubset({0,1,255}):raise ValueError('Unexpected annotation values')
        targets.append(mask>0)
    return image,np.stack(targets),np.asarray(available,dtype=bool)


def prepare_idrid(root,output,seed=42):
    root=Path(root).resolve();output=Path(output)
    if output.exists():raise FileExistsError('Preserve previous manifest')
    masks={}
    for p in root.rglob('*.tif'):
        if p.stem in masks:raise ValueError('Ambiguous annotation filename')
        masks[p.stem]=p
    rows=[];thumbs={}
    for path in sorted(root.rglob('*.jpg')):
        parts=path.relative_to(root).parts
        if '1. Original Images' not in parts:continue
        partition='train' if 'a. Training Set' in parts else 'test' if 'b. Testing Set' in parts else None
        if partition is None:raise ValueError('Unknown official partition')
        row={'sample_id':path.stem,'image':str(path.relative_to(root)),'official_split':partition,
             'image_sha256':sha256(path),'masks':{k:str(masks[path.stem+'_'+k].relative_to(root)) if path.stem+'_'+k in masks else None for k in CLASSES}}
        image,targets,available=read_idrid(root,row)
        row.update(width=image.width,height=image.height,available=available.tolist(),
                   positive_pixels=targets.sum(axis=(1,2)).tolist(),
                   mask_sha256={k:sha256(root/p) if p else None for k,p in row['masks'].items()})
        pixels=np.asarray(image);row['pixel_sha256']=hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest()
        small=np.asarray(image.convert('L').resize((9,8),Image.Resampling.BILINEAR))
        row['dhash']=f'{sum(int(b)<<i for i,b in enumerate((small[:,1:]>small[:,:-1]).flat)):016x}'
        thumbs[path.stem]=np.asarray(image.resize((64,64),Image.Resampling.BILINEAR),dtype=np.float32)/255
        rows.append(row)
    if not rows or len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Missing or ambiguous images')
    candidates=group_similar(rows,thumbs);groups=defaultdict(list)
    for row in rows:groups[row['similarity_group']].append(row)
    if any(len({r['official_split'] for r in g})>1 for g in groups.values()):raise ValueError('Official cross-split similarity requires review')
    ids=sorted(k for k,g in groups.items() if g[0]['official_split']=='train');random.Random(seed).shuffle(ids)
    val=set(ids[:max(1,round(.2*len(ids)))])
    for row in rows:row['split']='test' if row['official_split']=='test' else 'validation' if row['similarity_group'] in val else 'train'
    result={'protocol':'idrid-four-channel-v1','dataset':'IDRiD','classes':list(CLASSES),'seed':seed,
            'source':'https://zenodo.org/records/17219542','provenance':'Third-party redistribution of the original segmentation ZIP; not byte-compared with authenticated IEEE download',
            'license':'CC-BY-4.0','patient_identity_verified':False,'records':rows,
            'counts':dict(Counter(r['split'] for r in rows)),'similarity_candidates':candidates,
            'missing_mask_policy':'Unknown class supervision; excluded from loss and class metrics, not treated as negative'}
    write_json(output,result);return result


class IdridDataset(Dataset):
    def __init__(self,root,manifest,split,size=384,augment=False):
        self.root=Path(root);self.rows=[r for r in manifest['records'] if r['split']==split]
        self.size=size;self.augment=augment;self.cache={}
        if not self.rows:raise ValueError('Empty split')
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        if index not in self.cache:
            image,targets,available=read_idrid(self.root,self.rows[index]);ys=[]
            for target in targets:
                pixels,y,valid=letterbox(image,target,self.size);ys.append(y)
            self.cache[index]=(pixels,np.stack(ys),valid,available)
        image,targets,valid,available=(a.copy() for a in self.cache[index])
        if self.augment and random.random()<.5:
            image=image[:,::-1].copy();targets=targets[:,:,::-1].copy();valid=valid[:,::-1].copy()
        return {'image':torch.from_numpy(image.transpose(2,0,1)).float()/255,'mask':torch.from_numpy(targets),
                'valid':torch.from_numpy(valid[None]),'available':torch.from_numpy(available),'id':self.rows[index]['sample_id']}
