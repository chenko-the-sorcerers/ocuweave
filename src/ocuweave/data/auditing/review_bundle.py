"""Read-only development review, including component survival under resampling."""
import csv
import json
from pathlib import Path
import subprocess
import zipfile
import numpy as np
from PIL import Image
from ocuweave.data.auditing.annotation_geometry import source_pair, restore_mask, boundary
from ocuweave.data.auditing.quality_screen import quality_metrics, fit_thresholds, quality_flags
from ocuweave.data.preprocessing.geometry import letterbox
from ocuweave.evaluation.segmentation import binary_metrics
from ocuweave.training.engine import checked_manifest
from ocuweave.utils.provenance import sha256, write_json


def component_labels(mask):
    """Four-connected run-length labeling; components are pixels, not clinical lesions."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError('Expected a two-dimensional mask')
    labels = np.zeros(mask.shape, dtype=np.int32)
    parent = [0]
    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    previous = []
    for y, row in enumerate(mask):
        changes = np.diff(np.pad(row.astype(np.int8), (1, 1)))
        runs = []
        cursor = 0
        for left, right in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            label = len(parent)
            parent.append(label)
            while cursor < len(previous) and previous[cursor][1] <= left:
                cursor += 1
            for a, b, other in previous[cursor:]:
                if a >= right:
                    break
                if b > left:
                    parent[root(other)] = root(label)
            labels[y, left:right] = label
            runs.append((left, right, label))
        previous = runs
    roots = [root(i) for i in range(len(parent))]
    dense = {value: i for i, value in enumerate(sorted(set(roots)))}
    mapping = np.array([dense[value] for value in roots], dtype=np.int32)
    return mapping[labels]


def resize_findings(image, mask, size, labels=None):
    labels = component_labels(mask) if labels is None else labels
    pixels, resized, valid = letterbox(image, mask, size)
    restored = restore_mask(resized, valid, image.size)
    yy, xx = np.nonzero(valid)
    height, width = int(yy.max()-yy.min()+1), int(xx.max()-xx.min()+1)
    sampled = np.asarray(Image.fromarray(labels).resize((width, height), Image.Resampling.NEAREST))
    ids = np.unique(labels); ids = ids[ids > 0]
    surviving = np.unique(sampled); surviving = surviving[surviving > 0]
    original = int(mask.sum())
    return {
        'size': size, 'content_width': width, 'content_height': height,
        'padding_left': int(xx.min()), 'padding_top': int(yy.min()),
        'padding_right': int(size-xx.max()-1), 'padding_bottom': int(size-yy.max()-1),
        'padding_fraction': float(1-valid.mean()),
        'mask_pixels_in_padding': int((resized.astype(bool) & ~valid.astype(bool)).sum()),
        'original_lesion_pixels': original, 'resized_lesion_pixels': int(resized.sum()),
        'original_area_fraction': float(mask.mean()),
        'resized_content_area_fraction': float(resized.sum()/valid.sum()),
        'small_union_le_1pct': bool(0 < mask.mean() <= .01),
        'roundtrip_dice': binary_metrics(restored, mask)['dice'],
        'roundtrip_iou': binary_metrics(restored, mask)['iou'],
        'exact_boundary_dice': binary_metrics(boundary(restored), boundary(mask))['dice'],
        'relative_area_error': (int(restored.sum())-original)/original if original else None,
        'whole_mask_vanished': bool(original and not resized.any()),
        'original_components_4connected': len(ids),
        'components_lost_by_sampling': int(len(np.setdiff1d(ids, surviving))),
        'resized_components_4connected': int(component_labels(resized).max()),
    }


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)


def draw_panel(image, mask, sizes, output, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1+len(sizes), figsize=(13, 8), layout='constrained')
    axes[0, 0].imshow(image); axes[0, 0].set_title('Original RGB; unchanged')
    axes[1, 0].imshow(mask, cmap='gray', vmin=0, vmax=1); axes[1, 0].set_title('Reference union mask')
    for col, size in enumerate(sizes, 1):
        pixels, resized, valid = letterbox(image, mask, size)
        axes[0, col].imshow(pixels)
        if resized.any(): axes[0, col].contour(resized, levels=[.5], colors=['cyan'], linewidths=.7)
        axes[0, col].set_title(f'{size} px input + reference; black padding')
        back = restore_mask(resized, valid, image.size)
        axes[1, col].imshow(image)
        if mask.any(): axes[1, col].contour(mask, levels=[.5], colors=['lime'], linewidths=.7)
        if back.any(): axes[1, col].contour(back, levels=[.5], colors=['magenta'], linewidths=.7)
        axes[1, col].set_title('Original green / roundtrip magenta')
    for ax in axes.flat: ax.axis('off')
    fig.suptitle(title+' | reference geometry only; no predictions', fontsize=10)
    fig.savefig(output, dpi=120); plt.close(fig)


def build_review(manifest, archive, output, *, allow_unverified=False, panel_count=8):
    output = Path(output)
    if output.exists(): raise FileExistsError('Preserve previous review artifacts')
    if panel_count < 0: raise ValueError('Panel count must be nonnegative')
    data = checked_manifest(manifest, archive, allow_unverified)
    rows = [r for r in data['records'] if r['split'] in {'train', 'validation'}]
    quality, findings, failures = [], [], []
    with zipfile.ZipFile(archive) as source:
        for row in rows:
            try:
                image, mask, intermediate = source_pair(source, row)
            except (ValueError, KeyError, OSError) as error:
                failures.append({'sample_id': row['sample_id'], 'reason': type(error).__name__})
                continue
            quality.append({'sample_id': row['sample_id'], 'split': row['split'],
                            'intermediate_mask_pixels': intermediate, **quality_metrics(image)})
            labels = component_labels(mask)
            for size in (192, 384):
                findings.append({'sample_id': row['sample_id'], 'split': row['split'],
                                 **resize_findings(image, mask, size, labels)})
    if not quality: raise ValueError('No readable development pairs')
    thresholds = fit_thresholds(quality)
    for row in quality:
        flags = quality_flags(row, thresholds)
        row.update(flags, flag_count=sum(flags.values()))
    ranked = sorted(quality, key=lambda r: (-r['flag_count'], r['sample_id']))
    small = sorted([r for r in findings if r['size']==192 and r['original_lesion_pixels']>0],
                   key=lambda r: (r['original_area_fraction'], r['sample_id']))
    controls = sorted(quality, key=lambda r: (r['flag_count'], r['sample_id']))
    candidates = ranked[:3] + small[:3] + controls[:2] + ranked[3:]
    selected = list(dict.fromkeys(r['sample_id'] for r in candidates))[:panel_count]
    output.mkdir(parents=True)
    lookup = {r['sample_id']: r for r in rows}
    with zipfile.ZipFile(archive) as source:
        for number, sample_id in enumerate(selected):
            image, mask, _ = source_pair(source, lookup[sample_id])
            draw_panel(image, mask, (192, 384), output/f'overlay-{number:03d}.png', f'Review example {number:03d}')
    panel_lookup = {sample_id: f'overlay-{n:03d}.png' for n, sample_id in enumerate(selected)}
    review = []
    q_lookup = {r['sample_id']: r for r in quality}
    for row in rows:
        q = q_lookup.get(row['sample_id'])
        review.append({'sample_id': row['sample_id'], 'split': row['split'],
            'technical_pair_check': 'passed' if q else 'failed_requires_investigation',
            'quality_flags': ';'.join(k for k, v in (q or {}).items() if k.endswith('_proxy') and v),
            'overlay_file': panel_lookup.get(row['sample_id'], ''),
            **{k:'unreviewed' for k in ('visual_alignment', 'boundary_completeness', 'blur',
               'illumination', 'artifacts', 'retina_cutoff', 'clinical_lesion_correctness')},
            'reviewer':'', 'reviewer_role':'', 'review_date':'', 'notes':'',
            'disposition':'retain_pending_review'})
    write_csv(output/'image-mask-review.csv', review)
    write_csv(output/'quality-metrics.csv', quality)
    write_csv(output/'resize-comparison.csv', findings)
    summary = {'protocol':'development-review-bundle-v1', 'manifest_sha256':sha256(manifest),
        'archive_sha256':data['archive_sha256'], 'development_samples':len(rows),
        'readable_pairs':len(quality), 'pair_failures':failures,
        'images_with_quality_flags':sum(r['flag_count']>0 for r in quality),
        'thresholds_fitted_on':'training only', 'quality_thresholds':thresholds,
        'panels_created':len(selected), 'panel_selection':'3 highest flag counts, 3 smallest positive unions, 2 lowest flag counts; deduplicated then filled by flag rank',
        'test_images_read':False, 'training_performed':False, 'automatic_exclusions':0,
        'annotations_changed':False, 'clinical_review_complete':False,
        'by_size':{}}
    for size in (192, 384):
        positives = [r for r in findings if r['size']==size and r['original_lesion_pixels']>0]
        small_rows = [r for r in positives if r['small_union_le_1pct']]
        summary['by_size'][str(size)] = {
            'positive_images':len(positives), 'small_positive_images':len(small_rows),
            'mean_positive_roundtrip_dice':float(np.mean([r['roundtrip_dice'] for r in positives])) if positives else None,
            'mean_small_roundtrip_dice':float(np.mean([r['roundtrip_dice'] for r in small_rows])) if small_rows else None,
            'whole_masks_vanished':sum(r['whole_mask_vanished'] for r in positives),
            'components_lost_by_sampling':sum(r['components_lost_by_sampling'] for r in positives)}
    summary['limitations'] = [
        'Dimensional agreement does not establish semantic alignment or clinical correctness.',
        'Quality proxies prioritize review; edge contact does not prove a cut-off retina.',
        'Connected components can include JPEG threshold artifacts; they are not lesion instances.',
        'Roundtrip metrics quantify preprocessing only; exact boundary Dice has zero pixel tolerance.',
        'Area fractions use the full original image, not a clinically segmented retinal field.',
        'Private per-image CSVs and overlays must not be committed to the public repository.']
    try: summary['source_commit'] = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError): summary['source_commit'] = 'unavailable'
    write_json(output/'summary.json', summary)
    (output/'README.md').write_text('# Development photo-mask review\n\n'
        'No source data, masks or splits were modified. All images remain retained.\n\n'
        'Review values: unreviewed / acceptable / concern / indeterminate. Enter reviewer role, date and evidence. '
        'Clinical fields require an ophthalmologist. Technical flags do not authorize exclusion.\n\n'
        'Panels are selected review examples, not a representative performance sample. '
        'See image-mask-review.csv for private sample-to-panel mapping.\n')
    return summary
