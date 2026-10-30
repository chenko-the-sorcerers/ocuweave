"""Deterministic QA panels and learning curves; saved only to the experiment output."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from ocuweave.data.segmentation import SegmentationDataset
from ocuweave.models.factory import make_model
from ocuweave.utils.provenance import write_json


@torch.no_grad()
def sample_panel(manifest, archive, output, checkpoint=None, split='train', count=6):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True) if checkpoint else None
    size = state['config']['size'] if state else 192
    dataset = SegmentationDataset(manifest, archive, split, size)
    # Fixed manifest order, not selected by performance. Include an empty image if available.
    positives = [i for i,r in enumerate(dataset.rows) if r['lesion_pixels']>0]
    negatives = [i for i,r in enumerate(dataset.rows) if r['lesion_pixels']==0]
    indices = (negatives[:1] + positives[:max(0,count-1)])[:count]
    if not indices:indices = list(range(min(count,len(dataset))))
    model = None
    if state:
        model = make_model(state['config']);model.load_state_dict(state['model']);model.eval()
    columns = 3 if model else 2
    fig, axes = plt.subplots(len(indices),columns,figsize=(columns*3.5,len(indices)*2.8),squeeze=False,layout='constrained')
    ids = []
    for row,index in enumerate(indices):
        sample = dataset[index];ids.append(sample['id'])
        image = sample['image'].permute(1,2,0).numpy()
        mask = sample['mask'][0].numpy()
        axes[row,0].imshow(image);axes[row,0].set_title(sample['id'],fontsize=10)
        axes[row,1].imshow(mask,cmap='gray',vmin=0,vmax=1);axes[row,1].set_title('Reference lesion mask',fontsize=10)
        if model:
            prediction = model(sample['image'].unsqueeze(0)).sigmoid()[0,0].numpy()>=.5
            axes[row,2].imshow(prediction,cmap='gray',vmin=0,vmax=1);axes[row,2].set_title('Predicted mask (p ≥ 0.5)',fontsize=10)
        for axis in axes[row]:axis.axis('off')
    fig.suptitle(f'OTFID v3 · {split} samples · image-level exploratory protocol',fontsize=12)
    fig.savefig(output,dpi=140,bbox_inches='tight');plt.close(fig)
    write_json(Path(output).with_suffix('.json'),{'sample_ids':ids,'split':split,'size':size,
        'selection':'first empty and first positive images in frozen manifest order, not ranked by model score',
        'source':'https://zenodo.org/records/13883485','license':'CC-BY-4.0','clinical_use':False})


def learning_curves(runs, output):
    fig,axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    palette = {'query':'#2457A7','fcn':'#B56B14'}
    for label,directory in runs.items():
        history=json.loads((Path(directory)/'history.json').read_text())
        epochs=[r['epoch'] for r in history]
        axes[0].plot(epochs,[r['train_loss'] for r in history],label=label,color=palette[label],linestyle='-' if label=='query' else '--')
        axes[1].plot(epochs,[r['validation']['positive_macro_dice'] for r in history],label=label,color=palette[label],linestyle='-' if label=='query' else '--')
    for axis in axes:
        axis.set_xlabel('Epoch');axis.grid(axis='y',alpha=.2);axis.legend(frameon=False)
        axis.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Training BCE + soft Dice loss')
    axes[1].set_ylabel('Validation Dice · positive images');axes[1].set_ylim(0,1)
    fig.suptitle('OTFID v3 pilot · same split, epochs and input size · seed 42',fontsize=12)
    fig.savefig(output,dpi=160,bbox_inches='tight');plt.close(fig)
