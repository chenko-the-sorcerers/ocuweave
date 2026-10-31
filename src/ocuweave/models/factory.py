"""Explicit architecture registry used by training and inference."""
from ocuweave.models.segmenter import OcuWeaveSegmenter
from ocuweave.baselines.compact_fcn import CompactFCN
from ocuweave.models.local_detail import LocalDetailSegmenter


def make_model(config):
    architectures = {'query': OcuWeaveSegmenter, 'fcn': CompactFCN, 'query_detail': LocalDetailSegmenter}
    if config['architecture'] not in architectures:
        raise ValueError('Unknown model architecture')
    return architectures[config['architecture']](width=config['width'], dim=config['dim'], queries=config['queries'])
