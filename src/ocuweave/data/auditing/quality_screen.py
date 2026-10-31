"""Reproducible technical review proxies; none is a clinical gradability label."""
import numpy as np
from PIL import Image

METRICS=('sharpness','mean_intensity','contrast','clipped_bright_fraction','dark_fraction','edge_contact','quadrant_range')

def gray_array(image, size):
    image=image.convert('L') if isinstance(image,Image.Image) else Image.fromarray(np.asarray(image).squeeze().astype(np.uint8)).convert('L')
    return np.asarray(image.resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)/255


def quality_metrics(image):
    gray=gray_array(image,256)
    # A non-dark support proxy, not a segmented retinal field.
    support=gray>.04
    core=support[1:-1,1:-1]
    pixels=gray[support]
    lap=gray[:-2,1:-1]+gray[2:,1:-1]+gray[1:-1,:-2]+gray[1:-1,2:]-4*gray[1:-1,1:-1]
    border=np.concatenate((support[0],support[-1],support[1:-1,0],support[1:-1,-1]))
    quadrants=[gray[y:y+128,x:x+128] for y in (0,128) for x in (0,128)]
    return {'sharpness':float(lap[core].var()) if core.any() else 0.,
        'mean_intensity':float(pixels.mean()) if pixels.size else 0.,
        'contrast':float(np.quantile(pixels,.95)-np.quantile(pixels,.05)) if pixels.size else 0.,
        'clipped_bright_fraction':float((pixels>=.98).mean()) if pixels.size else 0.,
        'dark_fraction':float((gray<=.04).mean()),'edge_contact':float(border.mean()),
        'quadrant_range':float(np.ptp([q.mean() for q in quadrants])),
        'support_fraction':float(support.mean())}


def fit_thresholds(rows):
    training=[r for r in rows if r['split'] in {'train','training'}]
    if not training:raise ValueError('Training-only threshold fitting requires training rows')
    return {key:{'p10':float(np.quantile([r[key] for r in training],.1)),
                 'p90':float(np.quantile([r[key] for r in training],.9))} for key in METRICS}


def quality_flags(row, thresholds):
    # Strict inequalities avoid labelling every tied observation as an outlier.
    return {'low_sharpness_proxy':row['sharpness']<thresholds['sharpness']['p10'],
        'low_illumination_proxy':row['mean_intensity']<thresholds['mean_intensity']['p10'],
        'low_contrast_proxy':row['contrast']<thresholds['contrast']['p10'],
        'bright_clipping_proxy':row['clipped_bright_fraction']>thresholds['clipped_bright_fraction']['p90'],
        'large_dark_region_proxy':row['dark_fraction']>thresholds['dark_fraction']['p90'],
        'field_edge_contact_proxy':row['edge_contact']>max(.1,thresholds['edge_contact']['p90']),
        'uneven_illumination_or_artifact_proxy':row['quadrant_range']>thresholds['quadrant_range']['p90']}


def descriptor(image):
    gray=gray_array(image,64)[3:-3,3:-3]
    gray=np.asarray(Image.fromarray(gray).resize((64,64),Image.Resampling.BILINEAR),dtype=np.float32)
    std=float(gray.std())
    if std<.01:return None
    return (gray-gray.mean())/std


def near_pairs(rows, descriptors, cosine=.995, fine_rmse=.10, block=128, limit=20000):
    """Cross-split integrity screen only. No patient identity or automatic grouping.

    Center-cropped grayscale normalization, 16x16 coarse cosine screen, then
    64x64 normalized RMSE. Identity, horizontal flip and 180-degree rotation.
    Unmatched crops/rotations/compression can be missed; findings need review.
    """
    indices=[i for i,d in enumerate(descriptors) if d is not None]
    if not indices:return {'pairs':[],'excluded_low_variance':len(rows),'truncated':False}
    fine=np.stack([descriptors[i] for i in indices])
    def coarse(array):
        x=array.reshape(len(array),16,4,16,4).mean(axis=(2,4)).reshape(len(array),-1)
        return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    reference=coarse(fine);found={};truncated=False
    for transform in ('identity','horizontal_flip','rotate_180'):
        transformed=fine if transform=='identity' else fine[:,:,::-1] if transform=='horizontal_flip' else fine[:,::-1,::-1]
        compact=coarse(transformed)
        for start in range(0,len(indices),block):
            scores=reference[start:start+block]@compact.T
            aa,bb=np.where(scores>=cosine)
            for a,b in zip(aa+start,bb):
                if b>=a:continue
                error=float(np.sqrt(np.mean((fine[a]-transformed[b])**2)))
                if error>fine_rmse:continue
                key=(int(a),int(b))
                if key in found and found[key]['normalized_rmse']<=error:continue
                if key not in found and len(found)>=limit:
                    truncated=True;break
                ra,rb=rows[indices[a]],rows[indices[b]]
                found[key]={'a':ra['sample_id'],'b':rb['sample_id'],'a_split':ra['split'],'b_split':rb['split'],
                    'cross_split':ra['split']!=rb['split'],'transform_of_b':transform,'normalized_rmse':error,
                    'coarse_cosine':float(scores[a-start,b]),'review_status':'unreviewed','same_patient_verified':False}
            if truncated:break
        if truncated:break
    return {'pairs':sorted(found.values(),key=lambda p:(not p['cross_split'],p['normalized_rmse'])),
        'excluded_low_variance':len(rows)-len(indices),'truncated':truncated,
        'thresholds':{'coarse_cosine':cosine,'fine_normalized_rmse':fine_rmse},
        'transform_scope':['identity','horizontal_flip','rotate_180'],
        'patient_independence_proven':False}
