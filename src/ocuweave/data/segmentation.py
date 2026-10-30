"""Read images directly from the immutable archive; never extract an entire ZIP."""
import json
import random
import zipfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from ocuweave.data.adapters.otfid import read_image
from ocuweave.data.preprocessing.geometry import letterbox


class SegmentationDataset(Dataset):
    def __init__(self, manifest, archive, split, size=192, augment=False):
        self.manifest=json.loads(Path(manifest).read_text())
        self.rows=[r for r in self.manifest['records'] if r['split']==split]
        if not self.rows:raise ValueError(f'Empty {split} split')
        self.archive=Path(archive);self.size=size;self.augment=augment
        self.cache={}

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        if index not in self.cache:
            with zipfile.ZipFile(self.archive) as archive:
                image=read_image(archive,row['image'])
                target=np.zeros((image.height,image.width),dtype=bool)
                for name in row['masks']:
                    values=np.asarray(read_image(archive,name,'L'))
                    target |= values > (0 if values.max()<=1 else 127)
            self.cache[index]=letterbox(image,target,self.size)
        image,target,valid=(a.copy() for a in self.cache[index])
        if self.augment and random.random()<.5:
            image=image[:,::-1].copy();target=target[:,::-1].copy();valid=valid[:,::-1].copy()
        return {'image':torch.from_numpy(image.transpose(2,0,1)).float()/255,
                'mask':torch.from_numpy(target[None]),'valid':torch.from_numpy(valid[None]),
                'id':row['sample_id'],'group':row['split_group']}
