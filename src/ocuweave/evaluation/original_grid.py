"""Comparable development metrics on the same original image grid at every input size."""
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import torch
from torch.nn import functional as F
from ocuweave.data.auditing.annotation_geometry import source_pair
from ocuweave.data.preprocessing.geometry import letterbox
from ocuweave.evaluation.diagnostics import require_development_split, area_bin, stratify
from ocuweave.evaluation.segmentation import binary_metrics, aggregate
from ocuweave.models.factory import make_model
from ocuweave.training.engine import checked_manifest, select_device
from ocuweave.utils.provenance import sha256, write_json


def restore_probabilities(probability, valid, original_shape):
    """Remove rectangular letterbox padding, then bilinear-resize probabilities.

    Threshold only after this transform. This is distinct from nearest-neighbor
    reference-mask roundtrip analysis.
    """
    valid = np.asarray(valid, dtype=bool)
    if probability.ndim != 4 or probability.shape[:2] != (1, 1):
        raise ValueError('Expected one B=1, C=1 probability map')
    if tuple(probability.shape[-2:]) != valid.shape or not valid.any():
        raise ValueError('A nonempty matching valid region is required')
    if len(original_shape) != 2 or min(original_shape) < 1:
        raise ValueError('Positive original height and width required')
    if not torch.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError('Finite probabilities in [0, 1] required')
    yy, xx = np.nonzero(valid)
    region = valid[yy.min():yy.max()+1, xx.min():xx.max()+1]
    if not region.all():
        raise ValueError('Letterbox valid region must be rectangular')
    cropped = probability[:, :, yy.min():yy.max()+1, xx.min():xx.max()+1]
    return F.interpolate(cropped, size=original_shape, mode='bilinear', align_corners=False)[0, 0]


@torch.no_grad()
def evaluate_original(manifest, archive, checkpoint, output, *, split='validation', device='cpu', allow_unverified=False):
    require_development_split(split)
    if Path(output).exists():raise FileExistsError('Preserve previous evaluation')
    data=checked_manifest(manifest,archive,allow_unverified)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['provenance']['manifest_sha256']!=sha256(manifest):raise ValueError('Checkpoint/manifest mismatch')
    dev=select_device(device);model=make_model(state['config']).to(dev)
    model.load_state_dict(state['model']);model.eval()
    size=state['config']['size'];rows=[]
    if dev.type=='cuda':torch.cuda.synchronize(dev)
    start=time.monotonic()
    with zipfile.ZipFile(archive) as source:
        for row in data['records']:
            if row['split']!=split:continue
            image,target,_=source_pair(source,row)
            pixels,_,valid=letterbox(image,target,size)
            x=torch.from_numpy(pixels.transpose(2,0,1).copy()).float()[None].to(dev)/255
            probability=model(x).sigmoid()
            restored=restore_probabilities(probability,valid,target.shape).cpu().numpy()
            m=binary_metrics(restored>=.5,target)
            rows.append({'sample_id':row['sample_id'],'source':row.get('source') or 'unknown',
                'label':row['label'],'area_bin':area_bin(float(target.mean())),
                'false_positive_area_fraction':m['fp']/target.size,**m})
    if not rows:raise ValueError('No development samples')
    if dev.type=='cuda':torch.cuda.synchronize(dev)
    healthy=[r['false_positive_area_fraction'] for r in rows if not r['target_positive']]
    result={'protocol':'otfid-original-grid-v1','split':split,'input_size':size,
        'evaluation_grid':'original image dimensions; bilinear restored probabilities, threshold 0.5',
        'checkpoint_sha256':sha256(checkpoint),'manifest_sha256':sha256(manifest),
        'checkpoint_epoch':state['epoch'],'metrics':aggregate(rows),
        'by_original_area':stratify(rows,'area_bin'),'by_source':stratify(rows,'source'),
        'by_label':stratify(rows,'label'),'mean_healthy_false_positive_area_fraction':float(np.mean(healthy)) if healthy else None,
        'elapsed_seconds_including_decode':time.monotonic()-start,'per_image':rows,
        'limitations':['Development-only; checkpoint selected using validation on its input grid.',
            'Original-grid metrics are not interchangeable with v0.2.0 input-grid metrics.',
            'Image bootstrap does not establish patient-level independence.']}
    write_json(output,result)
    return result
