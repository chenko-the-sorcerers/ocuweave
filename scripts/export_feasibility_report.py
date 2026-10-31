"""Export measured CPU feasibility receipts and licensed display panels, never checkpoints."""
import argparse
import json
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from ocuweave.utils.provenance import sha256, write_json, software_provenance


def panel(image, mask, path, color):
    image=image.convert('RGB'); mask=np.asarray(mask,dtype=bool)
    if mask.shape!=(image.height,image.width):raise ValueError('Display geometry mismatch')
    rgb=np.asarray(image).copy()
    rgb[mask]=np.rint(.35*rgb[mask]+.65*np.asarray(color)).astype(np.uint8)
    result=Image.fromarray(rgb);result.thumbnail((1000,1000),Image.Resampling.LANCZOS)
    result.save(path,quality=90)


def export_task(task,root,manifest_path,run_path,output):
    root,run_path,output=Path(root),Path(run_path),Path(output)
    manifest=json.loads(Path(manifest_path).read_text())
    results=json.loads((run_path/'results.json').read_text())
    if len(results)!=3 or any(r['steps_completed']!=100 for r in results):
        raise ValueError('Require all three completed 100-step feasibility runs')
    protocol=json.loads((run_path/'protocol.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    for path in (run_path/'protocol.json',run_path/'results.json'):
        (output/path.name).write_bytes(path.read_bytes())
    for run in results:
        name=run['run'];dst=output/name;dst.mkdir()
        for filename in ('completed.json','training-complete.json','history.json'):
            (dst/filename).write_bytes((run_path/name/filename).read_bytes())
        (dst/'metrics.json').write_bytes((run_path/name/'validation/metrics.json').read_bytes())
        if (run_path/name/'smoke.json').exists():
            (dst/'smoke.json').write_bytes((run_path/name/'smoke.json').read_bytes())
    evaluation=json.loads((run_path/'lqnet384/validation/metrics.json').read_text())
    rows={r['sample_id']:r for r in manifest['records']}
    examples=[]
    if task=='fives':
        from ocuweave.data.adapters.fives import read_pair
        ordered=sorted(evaluation['per_image'],key=lambda r:(r['dice'],r['sample_id']))
        choices=[('Lowest Dice',ordered[0]),('Median Dice',ordered[len(ordered)//2]),('Highest Dice',ordered[-1])]
        for index,(selection,metrics) in enumerate(choices):
            sid=metrics['sample_id'];row=rows[sid];image,mask=read_pair(root,row)
            pred=np.asarray(Image.open(run_path/'lqnet384/validation'/f'{sid}-prediction.png'))>0
            prefix=f'fives-{index+1}'
            photo=image.copy();photo.thumbnail((1000,1000),Image.Resampling.LANCZOS);photo.save(output/f'{prefix}-photo.jpg',quality=90)
            panel(image,mask,output/f'{prefix}-reference.jpg',(79,208,153))
            panel(image,pred,output/f'{prefix}-prediction.jpg',(255,183,77))
            examples.append({'sample_id':sid,'selection':selection+' in LQNet validation; ties sorted by ID','prefix':prefix,'metrics':metrics,'source_image_sha256':row['image_sha256'],'source_mask_sha256':row['mask_sha256']})
    else:
        from ocuweave.data.adapters.idrid import read_idrid,CLASSES
        sid=evaluation['gallery_sample_ids'][0];image,masks,available=read_idrid(root,rows[sid])
        photo=image.copy();photo.thumbnail((1000,1000),Image.Resampling.LANCZOS);photo.save(output/'idrid-photo.jpg',quality=90)
        for c,name in enumerate(CLASSES):
            pred=np.asarray(Image.open(run_path/'lqnet384/validation'/f'{sid}-{name}.png'))>0
            prefix=f'idrid-{name.lower()}'
            panel(image,masks[c],output/f'{prefix}-reference.jpg',(79,208,153))
            panel(image,pred,output/f'{prefix}-prediction.jpg',(255,183,77))
            scores=next((r for r in evaluation['per_image_class'][name] if r['sample_id']==sid),None)
            examples.append({'sample_id':sid,'class':name,'annotation_available':bool(available[c]),'selection':'First validation ID, fixed before training','prefix':prefix,'photo':'idrid-photo.jpg','metrics':scores,'source_image_sha256':rows[sid]['image_sha256'],'source_mask_sha256':rows[sid]['mask_sha256'][name]})
    summary={'dataset':task,'exporter':software_provenance(),
             'environment':json.loads((run_path/'environment.json').read_text()) if (run_path/'environment.json').exists() else None,
             'protocol':protocol,'results':results,'examples':examples,
             'manifest_sha256':sha256(manifest_path),'counts':manifest['counts'],
             'patient_identity_verified':False,'test_evaluated':False,
             'source_manifest_fields':{k:v for k,v in manifest.items() if k not in {'records','similarity_candidates'}},
             'display':'Full-field resized photographs; reference green and prediction amber at 65% opacity. Metrics use original binary masks, never display pixels.',
             'assets':{p.name:sha256(p) for p in output.glob('*.jpg')}}
    write_json(output/'summary.json',summary)
    with zipfile.ZipFile(output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file():archive.write(path,path.relative_to(output))
    return output.with_suffix('.zip')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['fives','idrid'],required=True)
    for arg in ('root','manifest','run','output'):parser.add_argument('--'+arg,type=Path,required=True)
    args=parser.parse_args()
    print(export_task(args.task,args.root,args.manifest,args.run,args.output))
