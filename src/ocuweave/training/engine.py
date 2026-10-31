"""Seeded training, validation-only checkpoint selection, and separate held-out evaluation."""
import json
import os
import platform
import random
import subprocess
import time
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ocuweave.data.segmentation import SegmentationDataset
from ocuweave.models.factory import make_model
from ocuweave.losses.segmentation import segmentation_loss
from ocuweave.evaluation.segmentation import binary_metrics,aggregate
from ocuweave.utils.provenance import sha256,write_json


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def select_device(name):
    if name=='auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    return torch.device(name)


def checked_manifest(manifest,archive,allow_unverified):
    data=json.loads(Path(manifest).read_text())
    if sha256(archive)!=data['archive_sha256']:raise ValueError('Archive differs from audited manifest')
    if not data['patient_identity_verified'] and not allow_unverified:
        raise ValueError('Exploratory image split requires explicit --allow-unverified-patients')
    ids=[r['sample_id'] for r in data['records']]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate sample identifiers')
    if {r['split'] for r in data['records']}!={'train','validation','test'}:raise ValueError('Three nonempty splits required')
    groups={};pixels={};patients={};similarity={}
    for r in data['records']:
        checks=[(r['split_group'],groups),(r['pixel_sha256'],pixels)]
        if r.get('similarity_group'):checks.append((r['similarity_group'],similarity))
        checks.extend((pid,patients) for pid in r.get('related_patient_ids',[]))
        if r.get('patient_id'):checks.append((r['patient_id'],patients))
        elif data['patient_identity_verified']:raise ValueError('Verified patient split has an unknown ID')
        for key,mapping in checks:
            if key in mapping and mapping[key]!=r['split']:raise ValueError('Cross-split duplicate/group leakage')
            mapping[key]=r['split']
    return data


@torch.no_grad()
def evaluate_loader(model,loader,device):
    model.eval();rows=[];total_loss=0;n=0
    for batch in loader:
        x,y,valid=(batch[k].to(device) for k in ('image','mask','valid'))
        logits=model(x);loss=segmentation_loss(logits,y,valid)
        total_loss+=loss.item()*len(x);n+=len(x)
        pred=logits.sigmoid().cpu().numpy()>=.5
        for i,sample_id in enumerate(batch['id']):
            rows.append({'sample_id':sample_id,'group':batch['group'][i],**binary_metrics(pred[i],
                batch['mask'][i].numpy(),batch['valid'][i].numpy())})
    metrics=aggregate(rows);metrics['loss']=total_loss/n
    return metrics,rows


def validate_config(config):
    required={'architecture','width','dim','queries','size','batch_size','epochs','learning_rate','seed'}
    if set(config)!=required:raise ValueError(f'Config requires exactly: {sorted(required)}')
    if config['architecture'] not in {'query','fcn','query_detail'}:raise ValueError('Unknown architecture')
    for key in ('width','dim','queries','size','batch_size','epochs'):
        if type(config[key]) is not int or config[key]<1:raise ValueError(f'{key} must be a positive integer')
    if config['width']<4 or config['dim']%4 or config['size']<16 or config['size']%16:
        raise ValueError('width >= 4, dim divisible by 4, size divisible by 16 required')
    if type(config['seed']) is not int or config['seed']<0:raise ValueError('Nonnegative integer seed required')
    if not 0<float(config['learning_rate'])<1:raise ValueError('Learning rate must be in (0,1)')


def train(manifest,archive,output,config,*,allow_unverified=False,device='auto'):
    validate_config(config)
    output=Path(output)
    if output.exists():raise FileExistsError('Use a new run directory; never overwrite previous results')
    data=checked_manifest(manifest,archive,allow_unverified)
    seed_everything(config['seed']);dev=select_device(device)
    if dev.type == 'cuda':
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    train_set=SegmentationDataset(manifest,archive,'train',config['size'],True)
    val_set=SegmentationDataset(manifest,archive,'validation',config['size'],False)
    generator=torch.Generator().manual_seed(config['seed'])
    train_loader=DataLoader(train_set,batch_size=config['batch_size'],shuffle=True,num_workers=0,generator=generator)
    val_loader=DataLoader(val_set,batch_size=config['batch_size'],shuffle=False,num_workers=0)
    model=make_model(config).to(dev)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config['epochs'])
    try:revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):revision='unavailable'
    provenance={'config':config,'source_commit':revision,'manifest_sha256':sha256(manifest),
        'archive_sha256':data['archive_sha256'],'split_unit':data['split_unit'],
        'device':str(dev),'torch':str(torch.__version__),'python':platform.python_version(),
        'parameters':sum(p.numel() for p in model.parameters()),'pretrained_weights':False,
        'device_name':torch.cuda.get_device_name(dev) if dev.type == 'cuda' else platform.processor(),
        'optimizer_steps_planned':len(train_loader)*config['epochs'],
        'clinical_inputs_used':False,'claim':'Exploratory image-only binary lesion segmentation'}
    output.mkdir(parents=True);write_json(output/'provenance.json',provenance)
    best=-1;history=[];start=time.monotonic();steps=0
    for epoch in range(1,config['epochs']+1):
        model.train();running=0;n=0
        for batch in train_loader:
            x,y,valid=(batch[k].to(dev) for k in ('image','mask','valid'))
            optimizer.zero_grad(set_to_none=True)
            loss=segmentation_loss(model(x),y,valid)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);optimizer.step()
            steps+=1
            running+=loss.item()*len(x);n+=len(x)
        validation,_=evaluate_loader(model,val_loader,dev)
        score=validation['positive_macro_dice']
        if score is None:score=validation['macro_dice']
        item={'epoch':epoch,'train_loss':running/n,'validation':validation,'elapsed_seconds':time.monotonic()-start}
        history.append(item);write_json(output/'history.json',history);print(json.dumps(item),flush=True)
        if score>best:
            best=score
            temporary=output/'best.tmp'
            torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'config':config,'epoch':epoch,'validation_score':score,'provenance':provenance},temporary)
            os.replace(temporary,output/'best.pt')
        scheduler.step()
    if dev.type == 'cuda':torch.cuda.synchronize(dev)
    try:
        import resource
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    except (ImportError, AttributeError):
        peak_rss=None
    write_json(output/'checkpoint.json',{'sha256':sha256(output/'best.pt'),'best_validation_positive_dice':best,
        'elapsed_seconds':time.monotonic()-start,'test_set_used_in_training':False,
        'optimizer_steps_completed':steps,'peak_process_rss_bytes':peak_rss,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(dev) if dev.type=='cuda' else None,
        'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(dev) if dev.type=='cuda' else None,
        'timing_scope':'Epoch loop including validation and checkpoint IO; excludes setup and final evaluation',
        'memory_scope':'CUDA since reset before model creation; CPU RSS is process lifetime high-water mark'})
    return output/'best.pt'


def evaluate(manifest,archive,checkpoint,output,*,split='test',allow_unverified=False,device='auto'):
    if split not in {'test','validation'}:raise ValueError('Held-out split required')
    if Path(output).exists():raise FileExistsError('Evaluation output already exists; preserve the original test result')
    data=checked_manifest(manifest,archive,allow_unverified)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['provenance']['manifest_sha256']!=sha256(manifest):raise ValueError('Checkpoint and manifest differ')
    dev=select_device(device);model=make_model(state['config']).to(dev);model.load_state_dict(state['model'])
    dataset=SegmentationDataset(manifest,archive,split,state['config']['size'])
    metrics,rows=evaluate_loader(model,DataLoader(dataset,batch_size=state['config']['batch_size']),dev)
    result={'split':split,'split_unit':data['split_unit'],'evaluation_resolution':state['config']['size'],
        'checkpoint_sha256':sha256(checkpoint),'checkpoint_epoch':state['epoch'],'metrics':metrics,'per_image':rows,
        'limitations':['Binary union of active/inactive lesions; no activity classification.',
            'No paired EMR, longitudinal endpoint or clinical validation in this experiment.',
            'Image-bootstrap intervals do not account for unknown patient linkage.']}
    write_json(output,result);return result
