"""Installed command-line interface; imports research dependencies only on use."""
import argparse
import json
from pathlib import Path

from ocuweave import __version__


def parser():
    root = argparse.ArgumentParser(prog='ocuweave', description='OcuWeave image-only segmentation research toolkit')
    root.add_argument('--version', action='version', version=f'OcuWeave {__version__}')
    commands = root.add_subparsers(dest='command', required=True)

    config = commands.add_parser('init-config', help='Write an explicit training configuration')
    config.add_argument('--output', required=True)
    config.add_argument('--architecture', choices=['query', 'query_detail', 'fcn'], default='query_detail')
    config.add_argument('--size', type=int, choices=[192, 384], default=192)
    config.add_argument('--seed', type=int, default=42)

    demo = commands.add_parser('demo', help='Train, evaluate and predict on generated geometric shapes (not retinal data)')
    demo.add_argument('--output', required=True, help='New directory; existing artifacts are never overwritten')
    demo.add_argument('--architecture', choices=['query', 'query_detail', 'fcn'], default='query_detail')
    demo.add_argument('--seed', type=int, default=42)
    demo.add_argument('--epochs', type=int, default=2)
    demo.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps', 'auto'])

    prepare = commands.add_parser('prepare-otfid', help='Audit an authorized OTFID ZIP and create a frozen manifest')
    prepare.add_argument('--archive', required=True)
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--seed', type=int, default=42)
    prepare.add_argument('--patient-map')
    prepare.add_argument('--image-column')
    prepare.add_argument('--label-column')
    prepare.add_argument('--allow-unverified-patients', action='store_true')

    for name in ('train', 'evaluate'):
        command = commands.add_parser(name, help=f'{name.capitalize()} using an audited manifest and local archive')
        for argument in ('manifest', 'archive', 'output'):
            command.add_argument('--' + argument, required=True)
        command.add_argument('--device', default='auto', choices=['cpu', 'cuda', 'mps', 'auto'])
        command.add_argument('--allow-unverified-patients', action='store_true')
        if name == 'train':
            command.add_argument('--config', required=True)
        else:
            command.add_argument('--checkpoint', required=True)
            command.add_argument('--split', choices=['validation', 'test'], default='validation')
            command.add_argument('--grid', choices=['original', 'input'], default='original',
                                 help='Original-grid evaluation is development-only; legacy test uses input grid')

    prediction = commands.add_parser('predict', help='Save original-size probability, binary mask and provenance')
    for argument in ('checkpoint', 'image', 'output'):
        prediction.add_argument('--' + argument, required=True)
    prediction.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps', 'auto'])
    return root


def dispatch(args):
    if args.command == 'init-config':
        if args.seed < 0:
            raise ValueError('Seed must be nonnegative')
        config = dict(architecture=args.architecture, width=24, dim=48, queries=8,
                      size=args.size, batch_size=8, epochs=20, learning_rate=0.001, seed=args.seed)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream:
            json.dump(config, stream, indent=2)
            stream.write('\n')
        return {'config': str(output)}
    if args.command == 'demo':
        from ocuweave.demo import run_demo
        return run_demo(args.output, architecture=args.architecture, seed=args.seed,
                        epochs=args.epochs, device=args.device)
    if args.command == 'prepare-otfid':
        from ocuweave.data.adapters.otfid import audit_otfid
        result = audit_otfid(args.archive, args.output, seed=args.seed,
                            allow_unverified_patients=args.allow_unverified_patients,
                            patient_map=args.patient_map, image_column=args.image_column,
                            label_column=args.label_column)
        return {key: result[key] for key in ('eligible_samples', 'splits', 'split_unit')}
    if args.command == 'train':
        from ocuweave.training.engine import train
        checkpoint = train(args.manifest, args.archive, args.output,
                           json.loads(Path(args.config).read_text()),
                           allow_unverified=args.allow_unverified_patients, device=args.device)
        return {'checkpoint': str(checkpoint)}
    if args.command == 'evaluate':
        if args.grid == 'original':
            if args.split != 'validation':
                raise ValueError('Original-grid CLI evaluation is validation-only. Keep test data out of development.')
            from ocuweave.evaluation.original_grid import evaluate_original as evaluate
        else:
            from ocuweave.training.engine import evaluate
        result = evaluate(args.manifest, args.archive, args.checkpoint, args.output,
                          split=args.split, allow_unverified=args.allow_unverified_patients, device=args.device)
        return {'evaluation_grid': args.grid, 'split': args.split, 'metrics': result['metrics']}
    if args.command == 'predict':
        from ocuweave.inference.segmentation import save_prediction
        return save_prediction(args.checkpoint, args.image, args.output, device=args.device)
    raise ValueError('Unsupported command')


def main(argv=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    try:
        result = dispatch(args)
    except ModuleNotFoundError as error:
        if error.name not in {'torch', 'numpy', 'PIL', 'matplotlib'}:
            raise
        command_parser.error('Research dependencies are missing. Install with: python -m pip install ".[research]" (from the repository).')
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        command_parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0
