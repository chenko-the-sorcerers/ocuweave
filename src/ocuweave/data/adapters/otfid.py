"""OTFID archive audit. Original pixels/labels are retained; unknown masks are excluded."""
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
import zipfile
import numpy as np
from PIL import Image
from ocuweave.utils.provenance import sha256, write_json
from ocuweave.data.splitting.groups import assign_splits


def read_image(archive, member, mode='RGB'):
    with archive.open(member) as stream:
        image = Image.open(stream)
        image.load()
    return image.convert(mode)


def mask_key(stem):
    # Only release-documented active suffix is removed; no arbitrary substring matching.
    return stem[:-2] if stem.lower().endswith('-a') else stem


def audit_otfid(archive_path, output, *, seed=42, allow_unverified_patients=False,
                patient_map=None, image_column=None, label_column=None):
    archive_path, output = Path(archive_path), Path(output)
    if output.exists():
        raise FileExistsError('Choose a new audit output directory; existing splits are immutable')
    patients = json.loads(Path(patient_map).read_text()) if patient_map else {}
    if not isinstance(patients,dict) or any(not isinstance(v,str) or not v.strip() for v in patients.values()):
        raise ValueError('Patient map must map image stems to nonempty verified ID strings')
    with zipfile.ZipFile(archive_path) as archive:
        names = [n for n in archive.namelist() if not n.endswith('/') and '__MACOSX' not in n]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP member names')
        if any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts for n in names):
            raise ValueError('Unsafe ZIP member path')
        csvs = [n for n in names if PurePosixPath(n).name.lower() == 'dataset_labels.csv']
        if len(csvs) != 1:
            raise ValueError('Expected exactly one dataset_labels.csv')
        raw_csv = archive.read(csvs[0]).decode('utf-8-sig')
        delimiter = ';' if raw_csv.splitlines()[0].count(';') > raw_csv.splitlines()[0].count(',') else ','
        labels = list(csv.DictReader(io.StringIO(raw_csv), delimiter=delimiter))
        columns = list(labels[0]) if labels else []
        if not image_column:
            image_column = next((c for c in columns if c.strip().lower() in {'image','filename','image_id','image name','file','image_name'}), None)
        if not label_column:
            label_column = next((c for c in columns if c.strip().lower() in {'label','class','diagnosis','type','status'}), None)
        if image_column not in columns or label_column not in columns:
            raise ValueError(f'Explicit --image-column and --label-column required; observed columns: {columns}')
        images = {PurePosixPath(n).stem:n for n in names if 'images' in [p.lower() for p in PurePosixPath(n).parts] and PurePosixPath(n).suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}}
        image_members = [n for n in names if 'images' in [p.lower() for p in PurePosixPath(n).parts] and PurePosixPath(n).suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}]
        if len(images)!=len(image_members):raise ValueError('Ambiguous image stems in archive')
        masks = defaultdict(list)
        for name in names:
            if 'masks' in [p.lower() for p in PurePosixPath(name).parts] and PurePosixPath(name).suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}:
                masks[mask_key(PurePosixPath(name).stem)].append(name)
        rows, exclusions, warnings = [], [], []
        thumbnails = {}
        seen_ids = set()
        for source in labels:
            stem = PurePosixPath(source[image_column].strip()).stem
            if stem in seen_ids:
                raise ValueError('Repeated label row; manual review required')
            seen_ids.add(stem)
            label = source[label_column].strip().lower().replace('_',' ').replace('-',' ')
            healthy = label in {'healthy','normal'}
            if stem not in images:
                exclusions.append({'sample_id':stem,'reason':'missing_image'});continue
            matched = sorted(masks.get(stem, []))
            if not matched and not healthy:
                exclusions.append({'sample_id':stem,'reason':'missing_mask_not_assumed_negative'});continue
            try:
                image = read_image(archive, images[stem])
                target = np.zeros((image.height, image.width), dtype=bool)
                for name in matched:
                    mask = read_image(archive, name, 'L')
                    if mask.size != image.size:
                        raise ValueError('mask_image_size_mismatch')
                    values = np.asarray(mask)
                    if not set(np.unique(values)).issubset({0,1,255}):
                        warnings.append({'sample_id':stem,'reason':'nonbinary_mask_thresholded_at_127'})
                    target |= values > (0 if values.max() <= 1 else 127)
                if healthy and target.any():
                    raise ValueError('healthy_label_with_positive_mask')
                if not healthy and not target.any():
                    raise ValueError('positive_label_with_empty_mask')
                thumbnails[stem] = np.asarray(image.resize((64,64),Image.Resampling.BILINEAR),dtype=np.float32)/255
                pixels = np.asarray(image)
                pixel_hash = hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest()
                thumbnail = np.asarray(image.convert('L').resize((9,8),Image.Resampling.BILINEAR))
                dhash = sum(int(bit)<<i for i,bit in enumerate((thumbnail[:,1:]>thumbnail[:,:-1]).flat))
                row = {'dhash':f'{dhash:016x}','sample_id':stem,'patient_id':patients.get(stem),'image':images[stem], 'masks':matched,
                    'source':source.get('Source'),'label':label,'width':image.width,'height':image.height,'lesion_pixels':int(target.sum()),
                    'mask_sha256':hashlib.sha256(target.tobytes()).hexdigest(),'pixel_sha256':pixel_hash,
                    'clinical_note':None,'clinical_features':{},'healthy_zero_mask':healthy and not matched}
                rows.append(row)
            except (OSError, ValueError) as exc:
                exclusions.append({'sample_id':stem,'reason':str(exc)})
        by_hash=defaultdict(list)
        for row in rows:by_hash[row['pixel_sha256']].append(row)
        retained=[]; duplicates=[]
        for group in by_hash.values():
            group.sort(key=lambda r:r['sample_id'])
            if len({r['mask_sha256'] for r in group})>1:
                exclusions.extend({'sample_id':r['sample_id'],'reason':'duplicate_pixels_conflicting_masks'} for r in group)
                continue
            group[0]['related_patient_ids']=sorted({r['patient_id'] for r in group if r['patient_id']})
            retained.append(group[0])
            duplicates.extend({'excluded':r['sample_id'],'retained':group[0]['sample_id']} for r in group[1:])
        retained.sort(key=lambda r:r['sample_id'])
        from ocuweave.data.auditing.similarity import group_similar
        near_duplicates=group_similar(retained,thumbnails)
        split_unit=assign_splits(retained,seed,allow_unverified_patients)
        manifest={'schema_version':2,'audit_protocol':'otfid-binary-v2','dataset':'OTFID','release':'3.0.0','archive_sha256':sha256(archive_path),
            'source_url':'https://zenodo.org/records/13883485','license':'CC-BY-4.0','seed':seed,
            'split_unit':split_unit,'patient_identity_verified':split_unit=='patient_group','records':retained}
        report={'source_labels':dict(Counter(r[label_column].strip().lower() for r in labels)),
            'audit_protocol':'otfid-binary-v2','source_label_rows':len(labels),'archive_images':len(images),'eligible_samples':len(retained),
            'labels':dict(Counter(r['label'] for r in retained)),'splits':dict(Counter(r['split'] for r in retained)),
            'exclusions':exclusions,'exact_duplicates_removed':duplicates,'warnings':warnings,
            'split_unit':split_unit,'paired_emr_available':False,'near_duplicate_candidates':near_duplicates,
            'similarity_policy':'Screen 64-bit dHash Hamming <= 4; group only if RGB 64x64 thumbnail RMSE <= 0.025 on [0,1]. Do not delete; not patient verification.',
            'unlabelled_images':sorted(set(images)-seen_ids),
            'limitations':['Patient identity absent unless supplied separately; image splits are exploratory.',
                'Exact duplicates removed; perceptual candidates grouped conservatively. Undetected duplicates and cross-view patient overlap remain possible.',
                'Active and inactive masks are unioned for binary lesion segmentation; activity is not predicted.']}
        output.mkdir(parents=True)
        write_json(output/'manifest.json',manifest);write_json(output/'audit.json',report)
        return report
