"""Deterministic grouped stratification. Unknown patient IDs are never invented."""
import random
from collections import defaultdict


def assign_splits(rows, seed=42, allow_unverified_patients=False):
    if not rows:
        raise ValueError('No eligible samples')
    unknown = any(not r.get('patient_id') for r in rows)
    if unknown and not allow_unverified_patients:
        raise ValueError('Patient IDs are unavailable. Use explicit exploratory image split or provide a patient map.')
    # Union patient groups and decoded-pixel duplicates before assigning any split.
    parent = list(range(len(rows)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    seen = {}
    for i, row in enumerate(rows):
        keys = [('pixels', row['pixel_sha256'])]
        if row.get('similarity_group'):keys.append(('similarity',row['similarity_group']))
        for patient in row.get('related_patient_ids',[]):keys.append(('patient',patient))
        if row.get('patient_id'):
            keys.append(('patient', row['patient_id']))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    strata = defaultdict(list)
    for group in groups.values():
        label = '|'.join(sorted({r['label'] for r in group}))
        strata[label].append(group)
    rng = random.Random(seed)
    for label in sorted(strata):
        group_list = sorted(strata[label], key=lambda g: min(r['sample_id'] for r in g))
        rng.shuffle(group_list)
        n = len(group_list)
        n_val = max(1, round(n * .15)) if n >= 7 else 0
        n_test = max(1, round(n * .15)) if n >= 7 else 0
        for i, group in enumerate(group_list):
            split = 'validation' if i < n_val else 'test' if i < n_val + n_test else 'train'
            for row in group:
                row['split'] = split
                row['split_group'] = min(r['sample_id'] for r in group)
    if set(r['split'] for r in rows) != {'train', 'validation', 'test'}:
        raise ValueError('Not enough independent groups for train, validation and test')
    return 'image_duplicate_group_exploratory' if unknown else 'patient_group'
