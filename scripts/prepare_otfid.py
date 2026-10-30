"""Audit OTFID and freeze a split; detailed exclusions stay in the audit JSON."""
import argparse
from collections import Counter
import json
from ocuweave.data.adapters.otfid import audit_otfid

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patient-map')
    parser.add_argument('--image-column')
    parser.add_argument('--label-column')
    parser.add_argument('--allow-unverified-patients', action='store_true',
                        help='Explicit exploratory image split; NOT patient-independent.')
    args = parser.parse_args()
    report = audit_otfid(args.archive, args.output, seed=args.seed,
        patient_map=args.patient_map, image_column=args.image_column,
        label_column=args.label_column, allow_unverified_patients=args.allow_unverified_patients)
    summary = {key: report[key] for key in ('source_label_rows','archive_images','eligible_samples','labels','splits','split_unit')}
    summary['exclusion_reasons'] = dict(Counter(row['reason'] for row in report['exclusions']))
    summary['exact_duplicates_removed'] = len(report['exact_duplicates_removed'])
    summary['similarity_candidates'] = len(report['near_duplicate_candidates'])
    summary['similarity_pairs_grouped'] = sum(pair['grouped'] for pair in report['near_duplicate_candidates'])
    print(json.dumps(summary,indent=2))
