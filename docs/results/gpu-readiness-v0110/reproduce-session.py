# GPU readiness diagnostics — independent session, software v0.11.0
from pathlib import Path
from datetime import datetime, timezone
import os, sys, json, subprocess, hashlib
import torch
from google.colab import drive
assert torch.cuda.is_available(), 'CUDA required: do not silently substitute CPU'
print({'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(0), 'utc': datetime.now(timezone.utc).isoformat()})
PROJECT = Path('/content/drive/MyDrive/OcuWeave — Datasets and experiments')
if not PROJECT.is_dir(): drive.mount('/content/drive')
assert PROJECT.is_dir()
print('Project available:', PROJECT.is_dir())
print('Raw dataset folders:', [p.name for p in (PROJECT/'01 — Raw datasets').iterdir()])
print('Audit folders:', [p.name for p in (PROJECT/'02 — Audits').iterdir()])

# Next cell

# Pin implementation; inspect only the two authorized dataset folders.
REVISION='0a855b2ade86473c84b60d8e59221ba667e642ba'
REPO=Path('/content/ocuweave-gpu-diagnostics-v0110')
if not REPO.exists(): subprocess.run(['git','clone','--branch','v0.11.0','https://github.com/chenko-the-sorcerers/ocuweave.git',str(REPO)],check=True,capture_output=True)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==REVISION
assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
STAMP=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
SESSION=PROJECT/'04 — Runs/GPU-readiness-v0.11.0'/STAMP
SESSION.mkdir(parents=True,exist_ok=False)
ENV=dict(os.environ,PYTHONPATH=str(REPO/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg')
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
for task in ['FIVES','IDRiD']:
    folder=PROJECT/'01 — Raw datasets'/task
    print(task,'raw:',[(p.name,p.stat().st_size if p.is_file() else 'directory') for p in folder.iterdir()])
    print(task,'manifests:',[str(p.relative_to(PROJECT)) for p in (PROJECT/'02 — Audits'/f'{task}-v0.10.0').glob('*/manifest.json')])
print('Session:',SESSION)
print('GPU:',subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip())

# Next cell

# Freeze budget before fitting; stage immutable source files on Colab scratch disk.
import shutil, zipfile, platform, importlib.metadata
sys.path.insert(0,str(REPO/'src'))
from ocuweave.training.task_config import default_config
from ocuweave.utils.provenance import write_json
CONFIG=dict(default_config(),steps=200,batch_size=4,flip_probability=0.,save_every=25)
CONTRACT={'protocol':'gpu-readiness-diagnostic-v1','source_commit':REVISION,'software_version':'0.11.0','config':CONFIG,'tasks':['idrid','fives'],'count':4,'selection':'Training-only lexical greedy positive-class coverage at input grid; then lexical fill. No predictions or held-out samples used.','pipeline_criteria':['finite loss/gradients','nonzero encoder/head/local updates','checkpoint reload matches'],'small_set_fit_criterion':'Each observed positive class: mean input-grid Dice on selected positive images >= 0.80. Fixed 200-step budget; no threshold tuning.','test_evaluated':False,'validation_evaluated':False,'main_matrix_started':False}
write_json(SESSION/'contract.json',CONTRACT)
write_json(SESSION/'config.json',CONFIG)
write_json(SESSION/'environment.json',{'python':platform.python_version(),'device':'cuda','gpu':torch.cuda.get_device_name(0),'precision':'float32','packages':{k:importlib.metadata.version(k) for k in ['torch','numpy','Pillow','scipy','scikit-image']}})
MANIFESTS={'idrid':PROJECT/'02 — Audits/IDRiD-v0.10.0/20261031T181542Z/manifest.json','fives':PROJECT/'02 — Audits/FIVES-v0.10.0/20261031T174615Z/manifest.json'}
EXPECTED={'idrid':'5efcc29e06b6bcb0a4d0a8cfd5a3600951485cbbef17f78ab10e8ed41d66a4d1','fives':'632b76bad18375cb824fc18761216a00428310fe806f4170c2f866199f13ad68'}
STAGE=Path('/content/ocuweave-readiness-data');STAGE.mkdir(exist_ok=False)
ROOTS={}
for task,folder,filename in [('idrid','IDRiD','A. Segmentation.zip'),('fives','FIVES','FIVES-official-v1.rar')]:
    assert digest(MANIFESTS[task])==EXPECTED[task]
    raw=PROJECT/'01 — Raw datasets'/folder/filename
    local=STAGE/filename
    print('Copying',task,raw.stat().st_size,'bytes',flush=True)
    shutil.copyfile(raw,local)
    destination=STAGE/task;destination.mkdir()
    if task=='idrid':
        with zipfile.ZipFile(local) as z:
            for member in z.infolist():
                assert (destination/member.filename).resolve().is_relative_to(destination.resolve())
            z.extractall(destination)
    else:
        extractor=shutil.which('unrar') or shutil.which('unrar-free')
        if not extractor:
            subprocess.run(['apt-get','update','-qq'],check=True,stdout=subprocess.DEVNULL)
            subprocess.run(['apt-get','install','-y','-qq','unrar-free'],check=True,stdout=subprocess.DEVNULL)
            extractor=shutil.which('unrar-free')
        subprocess.run([extractor,'x',str(local),str(destination)+'/'],check=True,stdout=subprocess.DEVNULL)
    record=json.loads(MANIFESTS[task].read_text())['records'][0]
    relative=Path(record['image']);candidates=[]
    for p in destination.rglob(relative.name):
        root=p
        for _ in relative.parts:root=root.parent
        if (root/relative)==p:candidates.append(root)
    assert len(candidates)==1,(task,candidates)
    ROOTS[task]=candidates[0]
    print('Staged',task,ROOTS[task],flush=True)
write_json(SESSION/'data-sources.json',{k:{'manifest':str(MANIFESTS[k]),'manifest_sha256':EXPECTED[k],'local_root':str(ROOTS[k])}for k in ROOTS})
print('Ready for integrity checks and training-only selection')

# Next cell

# Verify data, select training samples, and inspect target survival before any fitting.
import numpy as np, matplotlib.pyplot as plt
from IPython.display import display, Image as DisplayImage
from ocuweave.data.tasks import SegmentationTask
from ocuweave.data.preprocessing.geometry import letterbox
from ocuweave.training.tasks import prepare_batch
from ocuweave.training.task_config import task_model
from ocuweave.losses.observed import observed_components
DATA={}; SELECTED={}; GEOMETRY=[]; SMOKE={}
for task in ['idrid','fives']:
    print('Verifying',task,flush=True)
    data=SegmentationTask(task,MANIFESTS[task],ROOTS[task],allow_unverified=True);DATA[task]=data
    ds=data.dataset('train',384);ordered=sorted(range(len(ds)),key=lambda i:ds.rows[i]['sample_id'])
    chosen=[];covered=set()
    for i in ordered:
        item=ds[i];available=item.get('available',torch.ones(1,dtype=torch.bool))
        positive={c for c in range(len(data.classes)) if available[c] and item['mask'][c].sum()>0}
        if positive-covered:chosen.append(i);covered|=positive
        if len(covered)==len(data.classes) or len(chosen)==4:break
    chosen += [i for i in ordered if i not in chosen][:4-len(chosen)]
    assert len(chosen)==4
    SELECTED[task]=[ds.rows[i]['sample_id'] for i in chosen]
    for i in chosen:
        row=ds.rows[i];image,targets,available=data.original(row)
        for c,cls in enumerate(data.classes):
            item={'task':task,'sample_id':row['sample_id'],'class':cls,'available':bool(available[c]),'original_pixels':int(targets[c].sum()) if available[c] else None}
            for size in [192,384]:
                _,target,valid=letterbox(image,targets[c],size)
                item[str(size)+'_pixels']=int(target.sum()) if available[c] else None
            GEOMETRY.append(item)
    fig,axes=plt.subplots(2,2,figsize=(9,8),layout='constrained')
    for ax,i in zip(axes.flat,chosen):
        item=ds[i];rgb=item['image'].permute(1,2,0).numpy();mask=item['mask'].any(0).numpy()
        overlay=rgb.copy();overlay[mask]=.5*overlay[mask]+.5*np.array([.95,.66,.1])
        ax.imshow(overlay);ax.set_title(item['id']+' | reference overlay',fontsize=11);ax.axis('off')
    fig.suptitle(task.upper()+' training tensors · 384 px · technical review only',fontsize=13)
    fig.savefig(SESSION/(task+'-tensor-review.png'),dpi=160);plt.close(fig)
    display(DisplayImage(filename=str(SESSION/(task+'-tensor-review.png')),width=750))
    batch=prepare_batch(ds,chosen,'cuda');torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model=task_model(task,CONFIG).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    before={k:v.detach().clone() for k,v in model.named_parameters()}
    parts=observed_components(model(batch['image']),batch['mask'],batch['valid'],batch['available']);parts['total'].backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);assert norm>0
    opt.step();assert all(torch.isfinite(v).all() for v in model.parameters())
    SMOKE[task]={'loss':{k:float(v.detach()) for k,v in parts.items()},'gradient_norm':float(norm),'changed_parameters':sum(not torch.equal(before[k],v) for k,v in model.named_parameters()),'device':'cuda','sample_ids':SELECTED[task]}
    del model,opt,before,batch,parts;torch.cuda.empty_cache()
    print(task,SMOKE[task],flush=True)
write_json(SESSION/'selection.json',SELECTED);write_json(SESSION/'geometry.json',GEOMETRY);write_json(SESSION/'cuda-smoke.json',SMOKE)
print('Frozen selections:',SELECTED)
print('Geometry:',json.dumps(GEOMETRY,indent=2))

# Next cell

display(DisplayImage(filename=str(SESSION/'idrid-tensor-review.png'),width=580))
display(DisplayImage(filename=str(SESSION/'fives-tensor-review.png'),width=580))

# Next cell

# Run each frozen small-set diagnostic once; preserve failure receipts.
write_json(SESSION/'technical-review.json',{'review':'Displayed 384px reference overlays for all eight selected training images. No gross image-mask shift visible at this display scale; clinical boundaries not verified.','small_structure_limit':'IDRiD_04 MA has 1 target pixel at 192 and 6 at 384; SE coverage is one selected image. FIVES lexical subset is not disease-balanced.','clinical_review_complete':False,'quality_exclusions':0})
for task in ['idrid','fives']:
    command=[sys.executable,'-m','ocuweave','diagnose','--task',task,'--manifest',str(MANIFESTS[task]),'--source',str(ROOTS[task]),'--config',str(SESSION/'config.json'),'--output',str(SESSION/task),'--device','cuda','--count','4','--allow-unverified-patients','--sample-ids',*SELECTED[task]]
    print('Starting',task,'200 steps on Tesla T4',datetime.now(timezone.utc).isoformat(),flush=True)
    with (SESSION/(task+'.log')).open('x') as log:
        result=subprocess.run(command,cwd=REPO,env=ENV,stdout=log,stderr=subprocess.STDOUT)
    print(task,'exit',result.returncode,flush=True)
    print((SESSION/(task+'.log')).read_text()[-1600:],flush=True)
    result.check_returncode()
    report=json.loads((SESSION/task/'diagnostics.json').read_text())
    print(task,'completed:',report['steps_completed'],'loss:',report['first_loss'],'->',report['last_loss'],'seconds:',report['elapsed_seconds'],flush=True)
print('Both diagnostics finished; main training remains unopened pending review.')

# Next cell

# Independently recount saved original-grid masks; summarize training-only diagnostics.
from PIL import Image
REPORTS={}; SUMMARY=[]; GATES={}; HISTORIES={}
for task in ['idrid','fives']:
    report=json.loads((SESSION/task/'diagnostics.json').read_text());REPORTS[task]=report
    history=json.loads((SESSION/task/'history.json').read_text());HISTORIES[task]=history
    assert report['steps_completed']==200 and report['gpu_executed'] and report['checkpoint_reload_matches']
    for row in report['original_grid']:
        if not row['annotation_available']:continue
        folder=SESSION/task/row['sample_id'];name=row['class']
        p=np.asarray(Image.open(folder/(name+'-prediction.png')))>0;y=np.asarray(Image.open(folder/(name+'-reference.png')))>0
        tp=int(np.count_nonzero(p&y));fp=int(np.count_nonzero(p&~y));fn=int(np.count_nonzero(~p&y))
        assert (tp,fp,fn)==tuple(row['metrics'][k] for k in ['tp','fp','fn'])
        assert abs(2*tp/(2*tp+fp+fn)-row['metrics']['dice'])<1e-12
    for name in DATA[task].classes:
        final=[r for r in report['final'] if r['class']==name and r['annotation_available'] and r['reference_pixels']>0]
        original=[r for r in report['original_grid'] if r['class']==name and r['annotation_available'] and r['reference_pixels']>0]
        item={'task':task,'class':name,'positive_training_images':len(final),'input_dice':float(np.mean([r['metrics']['dice'] for r in final])),'original_dice':float(np.mean([r['metrics']['dice'] for r in original])),'original_iou':float(np.mean([r['metrics']['iou'] for r in original])),'input_foreground_probability_mean':float(np.mean([r['foreground_probability']['mean'] for r in final])),'original_tp':sum(r['metrics']['tp'] for r in original),'original_fp':sum(r['metrics']['fp'] for r in original),'original_fn':sum(r['metrics']['fn'] for r in original)}
        item['small_set_fit_pass']=item['input_dice']>=.8;SUMMARY.append(item)
    groups=sorted(history[0]['parameter_groups'])
    update_totals={g:sum(r['parameter_groups'][g]['update_l2'] for r in history) for g in groups}
    gradient_totals={g:sum(r['parameter_groups'][g]['gradient_l2'] for r in history) for g in groups}
    GATES[task]={'pipeline_pass':bool(all(v>0 for v in update_totals.values()) and all(v>0 for v in gradient_totals.values()) and report['checkpoint_reload_matches']),'small_set_fit_pass':all(r['small_set_fit_pass'] for r in SUMMARY if r['task']==task),'checkpoint_reload_matches':report['checkpoint_reload_matches'],'original_metrics_recount_matches':True,'update_l2_sums_by_group':update_totals,'gradient_l2_sums_by_group':gradient_totals,'steps':200,'elapsed_seconds':report['elapsed_seconds'],'peak_cuda_allocated_bytes':report['peak_cuda_allocated_bytes'],'peak_cuda_reserved_bytes':report['peak_cuda_reserved_bytes'],'first_loss':report['first_loss'],'last_loss':report['last_loss']}
write_json(SESSION/'summary.json',{'scope':'training-only diagnostics; not validation or generalization','source_commit':REVISION,'classes':SUMMARY,'gates':GATES,'test_evaluated':False,'validation_evaluated':False,'main_matrix_started':False})
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for column,task in enumerate(['idrid','fives']):
    history=HISTORIES[task];ax=axes[0,column]
    for key,label,color,style in [('bce','BCE','#3155a6','-'),('soft_dice','Soft Dice loss','#c17919','--'),('total','Total','#404040',':')]:
        ax.plot([r['step'] for r in history],[r[key] for r in history],label=label,color=color,linestyle=style)
    ax.set(xlabel='Optimizer step',ylabel='Loss',title=task.upper()+' · four training images',ylim=(0,1.25));ax.legend(frameon=False)
    rows=[r for r in SUMMARY if r['task']==task];x=np.arange(len(rows));ax=axes[1,column]
    ax.bar(x-.18,[r['input_dice'] for r in rows],.36,color='#3155a6',label='384 px grid')
    ax.bar(x+.18,[r['original_dice'] for r in rows],.36,color='#d9a03c',label='Original grid')
    ax.set(xticks=x,xticklabels=[r['class']+' (n='+str(r['positive_training_images'])+')' for r in rows],ylim=(0,1),ylabel='Mean positive-image Dice')
    ax.axhline(.8,color='#555555',linestyle=':',linewidth=1,label='Prespecified input-grid fit goal');ax.legend(frameon=False,fontsize=8)
fig.suptitle('LQNet GPU readiness diagnostics · Tesla T4 · seed 42 · 200 steps',fontsize=14)
fig.savefig(SESSION/'diagnostic-curves.png',dpi=180);plt.close(fig)
print(json.dumps({'classes':SUMMARY,'gates':{k:{x:v for x,v in d.items() if not x.endswith('_by_group')}for k,d in GATES.items()}},indent=2))
display(DisplayImage(filename=str(SESSION/'diagnostic-curves.png'),width=760))

# Next cell

# Label roundtrip is a geometry diagnostic, not a model or accuracy upper bound.
from ocuweave.evaluation.original_grid import restore_probabilities
ROUNDTRIP=[]
for task in ['idrid','fives']:
    data=DATA[task]
    for row in data.manifest['records']:
        if row['sample_id'] not in SELECTED[task]:continue
        assert row['split']=='train'
        image,targets,available=data.original(row)
        for c,name in enumerate(data.classes):
            if not available[c]:continue
            target=targets[c]
            for size in [192,384]:
                _,y,valid=letterbox(image,target,size)
                restored=restore_probabilities(torch.from_numpy(y)[None,None],valid,target.shape).numpy()>=.5
                tp=int(np.count_nonzero(restored&target));fp=int(np.count_nonzero(restored&~target));fn=int(np.count_nonzero(~restored&target))
                ROUNDTRIP.append({'task':task,'sample_id':row['sample_id'],'class':name,'size':size,'reference_pixels':int(target.sum()),'input_pixels':int(y.sum()),'tp':tp,'fp':fp,'fn':fn,'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.})
write_json(SESSION/'label-roundtrip.json',{'interpretation':'Nearest label resize followed by evaluator bilinear restoration and threshold 0.5; not a mathematical performance ceiling','rows':ROUNDTRIP})
ROUNDTRIP_SUMMARY=[]
for task in ['idrid','fives']:
    for name in DATA[task].classes:
        result={'task':task,'class':name}
        for size in [192,384]:
            rows=[r for r in ROUNDTRIP if r['task']==task and r['class']==name and r['size']==size and r['reference_pixels']>0]
            result[str(size)+'_roundtrip_dice']=float(np.mean([r['dice'] for r in rows]))
        ROUNDTRIP_SUMMARY.append(result)
print(json.dumps(ROUNDTRIP_SUMMARY,indent=2))
write_json(SESSION/'label-roundtrip-summary.json',ROUNDTRIP_SUMMARY)