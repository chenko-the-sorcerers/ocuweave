"""Locked exploratory protocol and CUDA smoke gate, separate from research scores."""
import json
import math
import os
from pathlib import Path
import subprocess
import time
import torch
from torch.utils.data import DataLoader
from ocuweave.data.segmentation import SegmentationDataset
from ocuweave.models.factory import make_model
from ocuweave.losses.segmentation import segmentation_loss
from ocuweave.training.engine import checked_manifest,seed_everything
from ocuweave.training.ablation import configurations,run_suite
from ocuweave.utils.provenance import sha256,write_json

PIPELINE={'input':'RGB float32 / 255','geometry':'aspect-preserving letterbox; bilinear image, nearest mask',
    'padding':'excluded from loss and metrics','augmentation':'paired horizontal flip p=0.5, training only',
    'loss':'masked BCEWithLogits + soft Dice','optimizer':'AdamW','weight_decay':1e-4,
    'scheduler':'cosine over 20 epochs','gradient_clip_norm':1.0,'precision':'float32',
    'initialization':'random; no pretrained weights; no smoke weights reused',
    'threshold':0.5,'small_lesion':'0 < union-mask area / original full image area <= 0.01',
    'checkpoint_selection':'best validation positive Dice on input grid',
    'evaluation':'validation only, restored probabilities on original image grid',
    'quality_policy':'retain all flagged images pending review','patient_independence_verified':False,
    'emr_training':False,'test_access':'metadata and protected hashes only; no test predictions'}


def revision():
    repo=Path(__file__).resolve().parents[3]
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():
        raise ValueError('Commit code before locking or running the protocol')
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()


def lock(manifest,archive,output):
    data=checked_manifest(manifest,archive,True)
    if data.get('development_revision',{}).get('protocol')!='otfid-development-grouping-v1':
        raise ValueError('New provisional-group development manifest required')
    configs=configurations();counts={split:sum(r['split']==split for r in data['records']) for split in ('train','validation','test')}
    contract={'protocol':'otfid-exploratory-v1','source_commit':revision(),
        'manifest_sha256':sha256(manifest),'archive_sha256':sha256(archive),'pipeline':PIPELINE,
        'configs':configs,'split_counts':counts,'optimizer_steps_per_run':math.ceil(counts['train']/8)*20,
        'test_evaluated':False,'all_models_without_unet':True}
    root=Path(output);root.mkdir(parents=True,exist_ok=True);path=root/'locked-protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=contract:raise ValueError('Locked protocol changed; preserve this suite')
    else:write_json(path,contract)
    for name,cfg in configs.items():
        target=root/(name+'.json')
        if target.exists() and json.loads(target.read_text())!=cfg:raise ValueError('Locked config changed')
        if not target.exists():write_json(target,cfg)
    run_suite(manifest,archive,root,device='cuda',execute=False,configs=configs)
    return contract


def smoke(manifest,archive,output,*,device='cuda'):
    root=Path(output)
    if root.exists():raise FileExistsError('Preserve every smoke attempt')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required; CPU is not a GPU smoke pass')
    if device not in ('cuda','cpu'):raise ValueError('Use explicit CPU or CUDA')
    data=checked_manifest(manifest,archive,True);commit=revision()
    root.mkdir(parents=True,exist_ok=False);records=[]
    torch.set_num_threads(2)
    for name,cfg in configurations().items():
        if cfg['seed']!=42:continue
        seed_everything(42);dev=torch.device(device)
        dataset=SegmentationDataset(manifest,archive,'train',cfg['size'],True)
        loader=DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=True,num_workers=0,
                          generator=torch.Generator().manual_seed(42))
        if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.monotonic();model=make_model(cfg).to(dev)
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=1e-4)
        losses=[];norms=[];preview=None;last_x=None
        for index,batch in enumerate(loader):
            if index==3:break
            x,y,valid=(batch[k].to(dev) for k in ('image','mask','valid'))
            if x.shape[0]!=8:raise ValueError('Smoke needs three full batches')
            assert x.shape[-2:]==y.shape[-2:]==valid.shape[-2:]
            assert torch.isfinite(x).all() and x.min()>=0 and x.max()<=1
            assert not (y.bool() & ~valid.bool()).any()
            optimizer.zero_grad(set_to_none=True);logits=model(x)
            assert logits.shape==y.shape
            loss=segmentation_loss(logits,y,valid)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite smoke loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Missing/nonfinite smoke gradients')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            if norm<=0:raise FloatingPointError('Zero overall gradient')
            optimizer.step();losses.append(float(loss.detach()));norms.append(float(norm));last_x=x.detach()
            if preview is None:preview={k:batch[k][:2] for k in ('image','mask','valid')}
        if len(losses)!=3:raise ValueError('Three batches required')
        model.eval()
        with torch.no_grad():expected=model(last_x).detach().cpu()
        checkpoint=root/(name+'-smoke.pt')
        torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':cfg,
            'purpose':'pipeline smoke only; never initialize research runs'},checkpoint)
        restored=make_model(cfg).to(dev)
        restored.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model']);restored.eval()
        with torch.no_grad():torch.testing.assert_close(restored(last_x).cpu(),expected)
        if device=='cuda':torch.cuda.synchronize()
        records.append({'variant':name,'batches':3,'losses':losses,'gradient_norms':norms,
            'checkpoint_sha256':sha256(checkpoint),'checkpoint_reload_matches':True,
            'elapsed_seconds':time.monotonic()-start,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else None})
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(7,7),layout='constrained')
        for i in range(2):
            image=preview['image'][i].permute(1,2,0).numpy();mask=preview['mask'][i,0].numpy()
            axes[i,0].imshow(image);axes[i,1].imshow(image)
            if mask.any():axes[i,1].contour(mask,levels=[.5],colors=['cyan'],linewidths=.6)
            for ax in axes[i]:ax.axis('off')
        axes[0,0].set_title('Augmented training image');axes[0,1].set_title('Paired reference-mask boundary')
        fig.suptitle(name+' · technical alignment, not clinical validation',fontsize=10)
        fig.savefig(root/(name+'-pairs.png'),dpi=120);plt.close(fig)
        del model,restored,optimizer,last_x,x,y,valid,logits,loss,expected
        if device=='cuda':torch.cuda.empty_cache()
    result={'status':'passed','purpose':'pipeline check, not research result','device':device,
        'gpu_gate_passed':device=='cuda','source_commit':commit,'manifest_sha256':sha256(manifest),
        'archive_sha256':data['archive_sha256'],'pipeline':PIPELINE,'runs':records,'test_evaluated':False,
        'torch':str(torch.__version__),'device_name':torch.cuda.get_device_name() if device=='cuda' else 'CPU'}
    write_json(root/'smoke.json',result);return result


def execute(manifest,archive,output,smoke_report):
    contract=lock(manifest,archive,output)
    report=json.loads(Path(smoke_report).read_text())
    if report.get('status')!='passed' or report.get('gpu_gate_passed') is not True:
        raise ValueError('A passed CUDA smoke test is required')
    for field in ('source_commit','manifest_sha256','archive_sha256','pipeline'):
        if report.get(field)!=contract[field]:raise ValueError('Smoke/protocol mismatch: '+field)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; suite remains pending')
    return run_suite(manifest,archive,output,device='cuda',execute=True,configs=contract['configs'])
