"""Foreground segmentation metrics; empty-mask behavior is explicit."""
import numpy as np


def binary_metrics(prediction,target,valid=None):
    prediction,target=np.asarray(prediction,dtype=bool),np.asarray(target,dtype=bool)
    if prediction.shape!=target.shape:raise ValueError('Shapes must match')
    if valid is not None:
        valid=np.asarray(valid,dtype=bool);prediction=prediction[valid];target=target[valid]
    tp=int((prediction & target).sum());fp=int((prediction & ~target).sum());fn=int((~prediction & target).sum())
    dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.0
    iou=tp/(tp+fp+fn) if tp+fp+fn else 1.0
    return {'tp':tp,'fp':fp,'fn':fn,'dice':dice,'iou':iou,'target_positive':bool(target.any()),
            'prediction_positive':bool(prediction.any())}


def aggregate(rows,seed=42):
    if not rows:raise ValueError('Cannot aggregate empty metrics')
    positive=[r for r in rows if r['target_positive']]
    empty=[r for r in rows if not r['target_positive']]
    tp,fp,fn=(sum(r[k] for r in rows) for k in ('tp','fp','fn'))
    values=np.array([r['dice'] for r in positive])
    # Each row is an image. This CI does NOT imply patient independence.
    ci=None
    if len(values)>1:
        rng=np.random.default_rng(seed)
        means=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(500)])
        ci=[float(v) for v in np.quantile(means,[.025,.975])]
    return {'samples':len(rows),'positive_samples':len(positive),'empty_samples':len(empty),
        'macro_dice':float(np.mean([r['dice'] for r in rows])),
        'positive_macro_dice':float(values.mean()) if len(values) else None,
        'macro_iou':float(np.mean([r['iou'] for r in rows])),
        'global_dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.0,
        'empty_image_false_positive_rate':float(np.mean([r['prediction_positive'] for r in empty])) if empty else None,
        'positive_dice_image_bootstrap_95ci':ci,'bootstrap_unit':'image (exploratory, not patient)',
        'threshold':.5,'empty_empty_dice':1.0}
