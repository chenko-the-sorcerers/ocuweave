"""IDRiD class-specific feasibility training with explicit annotation availability."""
import json
import time
import resource
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from ocuweave.data.adapters.idrid import CLASSES,IdridDataset,read_idrid
from ocuweave.data.preprocessing.geometry import letterbox
from ocuweave.evaluation.original_grid import restore_probabilities
from ocuweave.evaluation.segmentation import binary_metrics,aggregate
from ocuweave.losses.segmentation import segmentation_loss
from ocuweave.models.multilabel import MultiLabelLQNet
from ocuweave.training.engine import seed_everything
from ocuweave.utils.provenance import sha256,write_json,software_provenance


def observed_loss(logits,targets,valid,available):
    if logits.shape!=targets.shape or available.shape!=targets.shape[:2]:raise ValueError('Class supervision shape mismatch')
    if not available.any():raise ValueError('No annotated class in batch')
    spatial=valid.expand_as(targets)
    return segmentation_loss(logits[available][:,None],targets[available][:,None],spatial[available][:,None])


@torch.no_grad()
def evaluate_lesions(model,root,manifest,size,output,device):
    output=Path(output);output.mkdir(exist_ok=False);model.eval();records={k:[] for k in CLASSES}
    selected=sorted(r['sample_id'] for r in manifest['records'] if r['split']=='validation')[:2]
    for row in manifest['records']:
        if row['split']!='validation':continue
        image,targets,available=read_idrid(root,row)
        pixels,_,valid=letterbox(image,targets[0],size)
        x=torch.from_numpy(pixels.transpose(2,0,1).copy()).float()[None].to(device)/255
        probabilities=model(x).sigmoid();predictions=[]
        for c,name in enumerate(CLASSES):
            prob=restore_probabilities(probabilities[:,c:c+1],valid,targets[c].shape).cpu().numpy()
            pred=prob>=.5;predictions.append(pred)
            Image.fromarray(pred.astype(np.uint8)*255).save(output/(row['sample_id']+'-'+name+'.png'))
            if available[c]:
                metrics=binary_metrics(pred,targets[c]);metrics['false_positive_area_fraction']=metrics['fp']/targets[c].size
                records[name].append({'sample_id':row['sample_id'],**metrics})
        if row['sample_id'] in selected:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(4,3,figsize=(12,12),layout='constrained')
            for c,name in enumerate(CLASSES):
                axes[c,0].imshow(image);axes[c,0].set_title(name+' · photograph')
                axes[c,1].imshow(targets[c],cmap='gray',vmin=0,vmax=1);axes[c,1].set_title('Reference' if available[c] else 'Annotation unavailable')
                axes[c,2].imshow(predictions[c],cmap='gray',vmin=0,vmax=1);axes[c,2].set_title('Prediction')
                for ax in axes[c]:ax.axis('off')
            fig.suptitle(row['sample_id']+' · IDRiD validation · fixed first two IDs',fontsize=12)
            fig.savefig(output/(row['sample_id']+'-panel.png'),dpi=130);plt.close(fig)
    per_class={k:aggregate(v) if v else None for k,v in records.items()}
    valid_scores=[v['positive_macro_dice'] for v in per_class.values() if v and v['positive_macro_dice'] is not None]
    result={'protocol':'idrid-original-grid-v1','split':'validation','threshold':.5,'per_class':per_class,
            'mean_class_positive_dice':float(np.mean(valid_scores)) if valid_scores else None,
            'per_image_class':records,'unavailable_annotations':'excluded; never assumed background',
            'test_evaluated':False,'patient_identity_verified':False,
            'gallery_sample_ids':selected,'grid':'original pixels; bilinear probability restoration before threshold'}
    write_json(output/'metrics.json',result);return result


def run_idrid(root,manifest_path,output,steps=100,device='cpu'):
    if type(steps) is not int or steps<1:raise ValueError('Positive steps required')
    data=json.loads(Path(manifest_path).read_text())
    if data.get('protocol')!='idrid-four-channel-v1':raise ValueError('Expected IDRiD four-channel manifest')
    seen={}
    for row in data['records']:
        if sha256(Path(root)/row['image'])!=row['image_sha256']:raise ValueError('Image changed')
        for k,p in row['masks'].items():
            if p and sha256(Path(root)/p)!=row['mask_sha256'][k]:raise ValueError('Annotation changed')
        for key in (row['similarity_group'],row['pixel_sha256']):
            if key in seen and seen[key]!=row['split']:raise ValueError('Cross-split group')
            seen[key]=row['split']
    output=Path(output);output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);dev=torch.device(device)
    protocol={'protocol':'idrid-feasibility-v1','classes':list(CLASSES),'seed':42,'steps':steps,'batch_size':8,
              'learning_rate':.001,'weight_decay':.0001,'precision':'float32','device':device,'torch':str(torch.__version__),
              'loss':'BCE + soft Dice over observed image-class pairs only',
              'pipeline':'RGB/255; letterbox bilinear image, nearest mask; padding excluded; paired horizontal flip p=.5',
              'checkpoint_selection':'Fixed final step; no validation tuning','manifest_sha256':sha256(manifest_path),
              'purpose':'Supplementary single-seed feasibility; not core-programme completion',**software_provenance()}
    write_json(output/'protocol.json',protocol);results=[]
    for name,size,detail in [('query192',192,False),('query384',384,False),('lqnet384',384,True)]:
        run=output/name;run.mkdir();seed_everything(42)
        ds=IdridDataset(root,data,'train',size,True)
        if len(ds)<8:raise ValueError('At least eight training samples required')
        loader=DataLoader(ds,batch_size=8,shuffle=True,drop_last=True,num_workers=0,generator=torch.Generator().manual_seed(42))
        iterator=iter(loader);model=MultiLabelLQNet(local_detail=detail).to(dev)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps);history=[];start=time.monotonic()
        for step in range(1,steps+1):
            try:batch=next(iterator)
            except StopIteration:iterator=iter(loader);batch=next(iterator)
            x,y,valid,available=(batch[k].to(dev) for k in ('image','mask','valid','available'))
            optimizer.zero_grad(set_to_none=True);loss=observed_loss(model(x),y,valid,available)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            if norm<=0:raise FloatingPointError('Zero gradient')
            optimizer.step();scheduler.step();history.append({'step':step,'loss':float(loss.detach()),'gradient_norm':float(norm)})
            if step%25==0 or step==steps:
                write_json(run/'history.json',history);print(name,'step',step,'loss',round(history[-1]['loss'],5),flush=True)
        elapsed=time.monotonic()-start;checkpoint=run/'final.pt';model.eval()
        with torch.no_grad():expected=model(x).cpu()
        torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'local_detail':detail,'size':size,'protocol':protocol},checkpoint)
        restored=MultiLabelLQNet(local_detail=detail).to(dev)
        restored.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model']);restored.eval()
        with torch.no_grad():torch.testing.assert_close(expected,restored(x).cpu())
        del restored
        record={'run':name,'steps_completed':steps,'seed':42,'device':device,'input_size':size,
                'training_elapsed_seconds':elapsed,'checkpoint_sha256':sha256(checkpoint),'checkpoint_reload_matches':True,
                'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                'rss_scope':'process lifetime, includes preceding variants','parameters':sum(p.numel() for p in model.parameters())}
        write_json(run/'training-complete.json',record)
        evaluation=evaluate_lesions(model,root,data,size,run/'validation',dev)
        record['per_class']=evaluation['per_class'];record['mean_class_positive_dice']=evaluation['mean_class_positive_dice']
        write_json(run/'completed.json',record);results.append(record);write_json(output/'results.json',results)
        print(json.dumps(record),flush=True)
        del model,optimizer,ds,loader,iterator,batch,x,y,valid,available,loss,expected
    return results
