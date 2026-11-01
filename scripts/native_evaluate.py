"""Restore native probabilities to source dimensions and quantify annotated image areas.

Image operations use Pillow; all overlap calculations use Python standard library.
No clinical interpretation, physical area, or early-detection claim is produced.
"""
import argparse
import array
import json
import math
import struct
import sys
from pathlib import Path
from PIL import Image
from native_data import load_case, sha256


def read_probabilities(path):
    with path.open('rb') as f:
        if f.read(8) != b'OWPROB01':
            raise ValueError('Invalid prediction format')
        shape = struct.unpack('<4I', f.read(16))
        count = math.prod(shape)
        if not all(shape) or count > 150_000_000:
            raise ValueError('Invalid prediction extent')
        values = array.array('f')
        values.fromfile(f, count)
        if sys.byteorder != 'little':
            values.byteswap()
        if f.read(1) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError('Invalid probability payload')
    return shape, values


def overlap(pred, truth):
    a, b = pred.tobytes(), truth.tobytes()
    tp = sum(bool(x) and bool(y) for x, y in zip(a, b))
    positive, reference = sum(bool(v) for v in a), sum(bool(v) for v in b)
    fp, fn = positive - tp, reference - tp
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'dice': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
            'iou': tp / (tp + fp + fn) if tp + fp + fn else 1.,
            'reference_area_pixels': reference}


def evaluate(prediction, export_metadata, root, output, threshold=.5):
    if output.exists() or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('New output directory and threshold in [0,1] required')
    meta = json.loads(export_metadata.read_text())
    shape, values = read_probabilities(prediction)
    n, c, h, w = shape
    if (n, c, h, w) != (len(meta['samples']), len(meta['classes']), meta['size'], meta['size']):
        raise ValueError('Prediction/metadata mismatch')
    if any(s['crop_xyxy'] is not None for s in meta['samples']):
        raise ValueError('Patch diagnostics require full-image tiled inference before original-grid evaluation')
    output.mkdir(parents=True)
    rows = []
    for i, sample in enumerate(meta['samples']):
        image, targets = load_case(root, sample['source'], meta['classes'])
        geo = sample['letterbox']
        for j, name in enumerate(meta['classes']):
            channel = values[(i*c+j)*h*w:(i*c+j+1)*h*w]
            probability = Image.new('F', (w, h))
            probability.putdata(channel)
            probability = probability.crop((geo['left'], geo['top'], geo['left']+geo['width'], geo['top']+geo['height']))
            probability = probability.resize(image.size, Image.Resampling.BILINEAR)
            pred = Image.new('L', image.size)
            pred.putdata([255 if v >= threshold else 0 for v in probability.getdata()])
            area = sum(pred.histogram()[1:])
            row = {'sample_id': sample['source_id'], 'class': name, 'annotation_available': targets[j] is not None,
                   'predicted_area_pixels': area, 'image_area_pixels': image.width*image.height,
                   'predicted_image_area_percent': 100*area/(image.width*image.height),
                   'bbox_xyxy_exclusive': pred.getbbox(), 'physical_area_mm2': None,
                   'metrics': overlap(pred, targets[j]) if targets[j] is not None else None}
            pred.save(output / (sample['source_id'] + '-' + name + '-prediction.png'))
            rows.append(row)
    summary = {}
    for name in meta['classes']:
        positive = [r['metrics'] for r in rows if r['class']==name and r['metrics'] and r['metrics']['reference_area_pixels']>0]
        summary[name] = {'positive_annotated_images': len(positive),
                         'positive_macro_dice': sum(r['dice'] for r in positive)/len(positive) if positive else None,
                         'positive_macro_iou': sum(r['iou'] for r in positive)/len(positive) if positive else None}
    report = {'scope': 'training-only diagnostic', 'grid': 'original image', 'threshold': threshold,
              'prediction_sha256': sha256(prediction), 'export_metadata_sha256': sha256(export_metadata),
              'summary': summary, 'rows': rows, 'area_denominator': 'whole image, not verified retina/FOV',
              'early_detection_validated': False, 'clinical_diagnosis': None}
    (output/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threshold',type=float,default=.5)
    a=p.parse_args()
    print(json.dumps(evaluate(a.predictions,a.metadata,a.root,a.output,a.threshold)['summary'],indent=2))
