"""Development-only geometric mask review; never adjudicates clinical boundaries."""
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image
from ocuweave.data.adapters.otfid import read_image
from ocuweave.data.preprocessing.geometry import letterbox
from ocuweave.evaluation.segmentation import binary_metrics
from ocuweave.training.engine import checked_manifest
from ocuweave.utils.provenance import sha256, write_json


def source_pair(archive, row):
    image = read_image(archive, row['image'])
    if image.size != (row['width'], row['height']):
        raise ValueError('Image geometry differs from frozen manifest')
    mask = np.zeros((image.height, image.width), dtype=bool)
    intermediate = 0
    for name in row['masks']:
        values = np.asarray(read_image(archive, name, 'L'))
        if values.shape != mask.shape:
            raise ValueError('Image and mask dimensions disagree')
        intermediate += int(((values > 1) & (values < 255)).sum())
        mask |= values > (0 if values.max() <= 1 else 127)
    if int(mask.sum()) != row['lesion_pixels']:
        raise ValueError('Decoded mask differs from frozen manifest')
    return image, mask, intermediate


def restore_mask(mask, valid, original_size):
    yy, xx = np.nonzero(valid)
    cropped = mask[yy.min():yy.max()+1, xx.min():xx.max()+1]
    return np.asarray(Image.fromarray(cropped.astype(np.uint8)*255).resize(
        original_size, Image.Resampling.NEAREST)) > 0


def boundary(mask):
    padded = np.pad(mask, 1)
    interior = (padded[:-2, 1:-1] & padded[2:, 1:-1] &
                padded[1:-1, :-2] & padded[1:-1, 2:])
    return mask & ~interior


def review_annotations(manifest, archive, output, *, sizes=(192, 384), selected=(), allow_unverified=False):
    output = Path(output)
    if output.exists():raise FileExistsError('Preserve previous annotation reviews')
    data = checked_manifest(manifest, archive, allow_unverified)
    rows = [r for r in data['records'] if r['split'] in {'train', 'validation'}]
    known = {r['sample_id'] for r in rows}
    if not set(selected).issubset(known):raise ValueError('Review selections must belong to development data')
    records, failures, panels = [], [], []
    with zipfile.ZipFile(archive) as source:
        for row in rows:
            try:
                image, mask, intermediate = source_pair(source, row)
            except (ValueError, KeyError, OSError):
                failures.append({'sample_id':row['sample_id'], 'reason':'source_pair_or_manifest_mismatch'})
                continue
            restored = []
            for size in sizes:
                _, resized, valid = letterbox(image, mask, size)
                back = restore_mask(resized, valid, image.size)
                restored.append(back)
                records.append({'sample_id':row['sample_id'], 'split':row['split'], 'size':size,
                    'source':row.get('source'), 'label':row['label'],
                    'original_lesion_pixels':int(mask.sum()), 'resized_lesion_pixels':int(resized.sum()),
                    'roundtrip_dice':binary_metrics(back, mask)['dice'],
                    'exact_boundary_dice':binary_metrics(boundary(back), boundary(mask))['dice'],
                    'area_relative_error':float((int(back.sum())-int(mask.sum()))/int(mask.sum())) if mask.any() else None,
                    'vanished':bool(mask.any() and not resized.any()),
                    'intermediate_mask_pixels_across_files':intermediate})
            if row['sample_id'] in selected:panels.append((row['sample_id'], image, mask, restored))
    summary = {'protocol':'otfid-geometry-review-v1','manifest_sha256':sha256(manifest),
        'archive_sha256':data['archive_sha256'], 'development_samples':len(rows),
        'pair_failures':failures,'test_images_read':False,'clinical_boundary_review_complete':False,
        'selection':'Previously selected adverse validation examples; not representative',
        'by_size':{str(size):{'positive_samples':sum(r['size']==size and r['original_lesion_pixels']>0 for r in records),
            'mean_positive_roundtrip_dice':float(np.mean([r['roundtrip_dice'] for r in records if r['size']==size and r['original_lesion_pixels']>0])),
            'vanished':sum(r['vanished'] for r in records if r['size']==size)} for size in sizes},
        'limitations':['Dimensions and overlays cannot prove clinical annotation correctness.',
            'Roundtrip and exact-boundary Dice measure resampling distortion, not model performance.',
            'Intermediate JPEG mask values are thresholded as in the frozen pilot; no labels are changed.']}
    output.mkdir(parents=True)
    write_json(output/'per_image.json',records)
    if panels:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # Separate each sample to keep original boundary overlays readable in Colab.
        for number,(sample_id,image,mask,restored) in enumerate(panels):
            fig,axes=plt.subplots(1,1+len(sizes),figsize=(12,4),layout='constrained')
            for ax,overlay,title in zip(axes,[mask]+restored,['Original mask']+[f'{s} px roundtrip' for s in sizes]):
                ax.imshow(image)
                if overlay.any():ax.contour(overlay,levels=[.5],colors=['cyan'],linewidths=.8)
                ax.set_title(title);ax.axis('off')
            fig.suptitle(f'{sample_id} · cyan: reference boundary after stated transform · no prediction')
            fig.savefig(output/f'overlay-{number:02d}.png',dpi=130);plt.close(fig)
    write_json(output/'summary.json',summary)
    return summary
