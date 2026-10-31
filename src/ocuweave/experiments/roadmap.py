"""Generate a design register, never an executable training or results receipt."""
import csv
import json
from pathlib import Path

SEEDS = (42, 123, 2026)
VARIANTS = (('query192', 'query', 192), ('query384', 'query', 384),
            ('detail384', 'query_detail', 384))
TRACKS = {
    'otfid': {'dataset': 'OTFID v3', 'targets': ['lesion_union'],
              'primary_metric': 'positive_image_macro_dice',
              'readiness': 'Frozen development manifest; CPU smoke passed; CUDA gate pending',
              'source': 'https://zenodo.org/records/13883485'},
    'fives': {'dataset': 'FIVES', 'targets': ['vessel'],
              'primary_metric': 'vessel_macro_dice',
              'readiness': 'Acquisition, grouping, target audit and dataset adapter pending',
              'source': 'https://doi.org/10.1038/s41597-022-01564-3'},
    'idrid': {'dataset': 'IDRiD', 'targets': ['microaneurysm', 'haemorrhage', 'hard_exudate', 'soft_exudate'],
              'primary_metric': 'observed_class_macro_dice',
              'readiness': 'Acquisition, annotation availability audit and four-channel pipeline pending',
              'source': 'https://idrid.grand-challenge.org/Data/'},
    'refuge2': {'dataset': 'REFUGE2', 'targets': ['disc', 'cup'],
                'primary_metric': 'disc_cup_macro_dice',
                'readiness': 'Access, mask encoding, patient/device audit and two-channel pipeline pending',
                'source': 'https://arxiv.org/abs/2202.08994'},
}
# These objectives are proposed and are NOT implemented training capabilities.
CONDITIONAL = {'fives': 'centerline_consistency', 'idrid': 'class_balanced_loss',
               'refuge2': 'cup_containment_loss'}
CONDITIONS = (
    {'id': 'clean', 'operation': 'identity'},
    {'id': 'blur1', 'operation': 'gaussian_blur', 'sigma_original_pixels': 1.0},
    {'id': 'blur2', 'operation': 'gaussian_blur', 'sigma_original_pixels': 2.0},
    {'id': 'dim080', 'operation': 'multiply_rgb_clip_0_1', 'factor': .8},
    {'id': 'dim060', 'operation': 'multiply_rgb_clip_0_1', 'factor': .6},
    {'id': 'noise001', 'operation': 'gaussian_rgb_noise_clip_0_1', 'standard_deviation': .01},
    {'id': 'noise003', 'operation': 'gaussian_rgb_noise_clip_0_1', 'standard_deviation': .03},
)


def make_plan():
    """Nine preserved OTFID designs + 27 new core + nine conditional designs."""
    rows = []
    for track, info in TRACKS.items():
        for seed in SEEDS:
            for variant, architecture, size in VARIANTS:
                rows.append(_row(track, info, seed, variant, architecture, size, 'core', None))
        if track in CONDITIONAL:
            for seed in SEEDS:
                rows.append(_row(track, info, seed, 'detail384-'+CONDITIONAL[track],
                                 'query_detail', 384, 'conditional', CONDITIONAL[track]))
    return {'schema_version': 1, 'protocol': 'ocuweave-cross-task-design-v1',
            'purpose': 'Design register; not runnable training configs or measured results',
            'status_as_of': '2026-11-01', 'core_runs': 36, 'conditional_runs': 9,
            'completed_research_runs': 0, 'tracks': TRACKS, 'runs': rows,
            'evaluation_conditions': CONDITIONS,
            'evaluation_policy': 'Original-grid validation; masks unchanged; identical corruptions per image across paired variants; no tuning on test',
            'historical_suite': {'source_commit': 'afd6b45c16761fba8cfc214c921cae87652e47c9',
                'manifest_sha256': '56c4721a3063cfb5e5fa89e909fbb47494fccc6c4b9f1597d973e1b3c3fe941e',
                'policy': 'Continue the existing OTFID suite; do not recreate its split or lock'},
            'release_gate': ['source/data/config/checkpoint hashes', 'full step budget and completion receipt',
                'task-specific evaluator checks', 'original-grid per-class denominators',
                'compute/environment record', 'permitted and provenance-linked gallery images']}


def _row(track, info, seed, variant, architecture, size, family, objective):
    return {'run_id': f'{track}-{variant}-seed{seed}', 'track': track, 'family': family,
            'seed': seed, 'architecture': architecture, 'size': size,
            'targets': list(info['targets']), 'proposed_additional_objective': objective,
            'status': 'pending_gpu' if track == 'otfid' else 'planned',
            'executable_config': False, 'optimizer_steps': 500, 'batch_size': 8,
            'initialization': 'random', 'unet': False, 'emr': False,
            'paired_control': f'{track}-detail384-seed{seed}' if objective else None,
            'primary_metric': info['primary_metric'], 'metric_value': None,
            'checkpoint_sha256': None, 'source_commit': None, 'manifest_sha256': None,
            'elapsed_seconds': None, 'peak_cuda_allocated_bytes': None}


def write_plan(output):
    root = Path(output)
    if root.exists():
        raise FileExistsError('Preserve existing design registers; choose a new output directory')
    plan = make_plan()
    root.mkdir(parents=True, exist_ok=False)
    (root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    fields = ['run_id', 'track', 'family', 'seed', 'architecture', 'size', 'status',
              'optimizer_steps', 'primary_metric', 'metric_value']
    with (root/'runs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(plan['runs'])
    (root/'README.md').write_text(
        '# Cross-task experiment design\n\n36 core + 9 conditional planned runs; 0 completed. '
        'Nine core rows refer to the already frozen OTFID suite.\n\n'
        'This register does not acquire data, implement new losses, train models or verify patient identity. '
        'Null metrics mean unmeasured, not zero. See docs/protocols/cross_task_experiments.md.\n')
    return plan
