"""FIVES feasibility experiments, distinct from the prespecified GPU programme."""
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ocuweave.data.adapters.fives import FivesDataset, read_pair, verify_fives
from ocuweave.data.preprocessing.geometry import letterbox
from ocuweave.evaluation.original_grid import restore_probabilities
from ocuweave.evaluation.segmentation import binary_metrics, aggregate
from ocuweave.losses.segmentation import segmentation_loss
from ocuweave.models.factory import make_model
from ocuweave.training.engine import seed_everything
from ocuweave.utils.provenance import sha256, software_provenance, write_json


def vessel_metrics(prediction, target):
    """Original-grid overlap, centerline overlap, and reference-radius recall."""
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize
    prediction=np.asarray(prediction,dtype=bool); target=np.asarray(target,dtype=bool)
    result=binary_metrics(prediction,target)
    ps=skeletonize(prediction); ts=skeletonize(target)
    precision=float((ps & target).sum()/ps.sum()) if ps.any() else 0.
    sensitivity=float((ts & prediction).sum()/ts.sum()) if ts.any() else 0.
    result['cldice']=2*precision*sensitivity/(precision+sensitivity) if precision+sensitivity else float(not target.any() and not prediction.any())
    result['sensitivity']=result['tp']/(result['tp']+result['fn']) if target.any() else None
    result['precision']=result['tp']/(result['tp']+result['fp']) if prediction.any() else None
    result['false_positive_area_fraction']=result['fp']/target.size
    radius=distance_transform_edt(target)
    thin=ts & (radius<=2)
    result['thin_centerline_reference_pixels']=int(thin.sum())
    result['thin_centerline_recalled_pixels']=int((thin & prediction).sum())
    result['thin_centerline_recall']=float((thin & prediction).sum()/thin.sum()) if thin.any() else None
    return result


@torch.no_grad()
def evaluate_vessels(model, root, manifest, size, output, device):
    model.eval(); records=[]; output=Path(output); output.mkdir(exist_ok=False)
    start=time.monotonic()
    rows=[r for r in manifest['records'] if r['split']=='validation']
    for index,row in enumerate(rows):
        image,target=read_pair(root,row)
        pixels,_,valid=letterbox(image,target,size)
        x=torch.from_numpy(pixels.transpose(2,0,1).copy()).float()[None].to(device)/255
        probability=restore_probabilities(model(x).sigmoid(),valid,target.shape).cpu().numpy()
        prediction=probability>=.5
        metrics=vessel_metrics(prediction,target)
        records.append({'sample_id':row['sample_id'],'disease_stratum':row['disease_stratum'],**metrics})
        Image.fromarray(prediction.astype(np.uint8)*255).save(output/(row['sample_id']+'-prediction.png'))
        if (index+1)%20==0: print('Evaluated',index+1,'/',len(rows),flush=True)
    summary=aggregate(records)
    for key in ('cldice','sensitivity','precision','false_positive_area_fraction','thin_centerline_recall'):
        values=[r[key] for r in records if r[key] is not None]
        summary[key]=float(np.mean(values)) if values else None
    result={'protocol':'fives-original-grid-v1','split':'validation',
            'grid':'original image pixels; bilinear restored probabilities; threshold 0.5',
            'thin_definition':'Reference skeleton pixels with Euclidean distance to background <=2 original pixels',
            'metrics':summary,'per_image':records,'elapsed_seconds':time.monotonic()-start,
            'test_evaluated':False,'patient_identity_verified':False}
    write_json(output/'metrics.json',result)
    # Predeclared low/median/high overlap examples plus largest false-positive area.
    ordered=sorted(records,key=lambda r:(r['dice'],r['sample_id']))
    chosen=[ordered[0],ordered[len(ordered)//2],ordered[-1],max(records,key=lambda r:r['false_positive_area_fraction'])]
    selected=[]
    for item in chosen:
        if item['sample_id'] not in selected:selected.append(item['sample_id'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for number,sample_id in enumerate(selected):
        row=next(r for r in rows if r['sample_id']==sample_id)
        image,target=read_pair(root,row)
        pred=np.asarray(Image.open(output/(sample_id+'-prediction.png')))>0
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for ax,title in zip(axes,('Fundus photograph','Reference vessels','LQNet / query prediction')):
            ax.set_title(title,fontsize=11);ax.axis('off')
        axes[0].imshow(image);axes[1].imshow(target,cmap='gray',vmin=0,vmax=1);axes[2].imshow(pred,cmap='gray',vmin=0,vmax=1)
        score=next(r for r in records if r['sample_id']==sample_id)
        fig.suptitle(f"{sample_id} · validation · Dice {score['dice']:.3f} · IoU {score['iou']:.3f}",fontsize=11)
        fig.savefig(output/f'example-{number:02d}.png',dpi=160);plt.close(fig)
    write_json(output/'gallery.json',{'selection':'lowest, median, highest Dice; largest FP area; deduplicated',
                                    'sample_ids':selected,'source':'FIVES, Jin et al., 2022, CC BY 4.0'})
    return result


def run_pilot(root, manifest_path, output, *, steps=100, device='cpu', seed=42):
    """Three equal-budget runs; final step fixed in advance, no early stopping."""
    if type(steps) is not int or steps<1: raise ValueError('Positive steps required')
    if device not in {'cpu','cuda'}:raise ValueError('Use explicit cpu or cuda')
    manifest=json.loads(Path(manifest_path).read_text())
    if manifest.get('protocol')!='fives-development-v1':raise ValueError('Expected FIVES manifest')
    verify_fives(root,manifest)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); dev=torch.device(device)
    protocol={'protocol':'fives-feasibility-v1','purpose':'Supplementary feasibility experiment; not a completed core-programme run',
              'steps':steps,'batch_size':8,'seed':seed,'learning_rate':.001,'weight_decay':.0001,
              'loss':'valid-region BCE + soft Dice','augmentation':'paired horizontal flip p=0.5',
              'precision':'float32','checkpoint_selection':'fixed final step; validation evaluated once after training',
              'device':device,'device_name':torch.cuda.get_device_name() if device=='cuda' else platform.processor(),
              'torch':str(torch.__version__),'manifest_sha256':sha256(manifest_path),
              'initialization':'random, reset after smoke','test_evaluated':False,**software_provenance()}
    write_json(output/'protocol.json',protocol)
    results=[]
    for name,architecture,size in [('query192','query',192),('query384','query',384),('lqnet384','query_detail',384)]:
        cfg={'architecture':architecture,'width':24,'dim':48,'queries':8,'size':size,'seed':seed}
        run_dir=output/name;run_dir.mkdir()
        seed_everything(seed)
        ds=FivesDataset(root,manifest,'train',size,True)
        if len(ds)<8:raise ValueError('At least eight training images required')
        loader=DataLoader(ds,batch_size=8,shuffle=True,drop_last=True,num_workers=0,
                          generator=torch.Generator().manual_seed(seed))
        model=make_model(cfg).to(dev)
        batch=next(iter(loader));x,y,valid=(batch[k].to(dev) for k in ('image','mask','valid'))
        loss=segmentation_loss(model(x),y,valid);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if not torch.isfinite(loss) or norm<=0:raise FloatingPointError('Smoke failed')
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Missing or invalid gradient')
        write_json(run_dir/'smoke.json',{'finite_loss':float(loss),'gradient_norm':float(norm),'batch_size':len(x),'device':device})
        # Smoke initialization/data ordering are discarded before the real run.
        seed_everything(seed);model=make_model(cfg).to(dev)
        loader=DataLoader(ds,batch_size=8,shuffle=True,drop_last=True,num_workers=0,
                          generator=torch.Generator().manual_seed(seed)); iterator=iter(loader)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps)
        if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.monotonic(); history=[]
        for step in range(1,steps+1):
            try:batch=next(iterator)
            except StopIteration:iterator=iter(loader);batch=next(iterator)
            x,y,valid=(batch[k].to(dev) for k in ('image','mask','valid'))
            model.train();optimizer.zero_grad(set_to_none=True)
            loss=segmentation_loss(model(x),y,valid)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step()
            history.append({'step':step,'loss':float(loss.detach()),'gradient_norm':float(norm)})
            if step%25==0 or step==steps:
                write_json(run_dir/'history.json',history)
                print(name,'step',step,'loss',round(history[-1]['loss'],5),'seconds',round(time.monotonic()-start),flush=True)
        if device=='cuda':torch.cuda.synchronize()
        elapsed=time.monotonic()-start
        checkpoint=run_dir/'final.pt';model.eval()
        with torch.no_grad():expected=model(x).cpu()
        torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':cfg,'protocol':protocol},checkpoint)
        restored=make_model(cfg).to(dev);restored.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model']);restored.eval()
        with torch.no_grad():torch.testing.assert_close(expected,restored(x).cpu())
        del restored
        evidence={'run':name,'steps_completed':steps,'checkpoint_sha256':sha256(checkpoint),
                  'checkpoint_reload_matches':True,'parameters':sum(p.numel() for p in model.parameters()),
                  'training_elapsed_seconds':elapsed,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
                  'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                  'rss_scope':'Process lifetime high-water mark; includes preceding variants',
                  'device':device,'seed':seed,'config':cfg}
        write_json(run_dir/'training-complete.json',evidence)
        metrics=evaluate_vessels(model,root,manifest,size,run_dir/'validation',dev)
        evidence['metrics']=metrics['metrics'];write_json(run_dir/'completed.json',evidence)
        results.append(evidence);write_json(output/'results.json',results)
        print(json.dumps(evidence),flush=True)
        del model,optimizer,ds,loader,iterator,batch,x,y,valid,loss,expected
    return results
