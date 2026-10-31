"""Four-channel IDRiD checkpoint inference on original image grids."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ocuweave.data.adapters.idrid import CLASSES
from ocuweave.data.preprocessing.geometry import letterbox
from ocuweave.evaluation.original_grid import restore_probabilities
from ocuweave.models.multilabel import MultiLabelLQNet
from ocuweave.utils.provenance import sha256,write_json


@torch.no_grad()
def predict_lesions(checkpoint,image,device='cpu'):
    """Return float probabilities shaped (4, original height, original width)."""
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    protocol=payload.get('protocol',{})
    if protocol.get('protocol')!='idrid-feasibility-v1' or protocol.get('classes')!=list(CLASSES):
        raise ValueError('Expected the four-channel IDRiD checkpoint contract')
    if type(payload.get('local_detail')) is not bool or payload.get('size') not in {192,384}:
        raise ValueError('Unknown IDRiD model configuration')
    model=MultiLabelLQNet(local_detail=payload['local_detail']).to(device)
    model.load_state_dict(payload['model'],strict=True);model.eval()
    image=image.convert('RGB');shape=(image.height,image.width)
    pixels,_,valid=letterbox(image,np.zeros(shape,dtype=bool),payload['size'])
    x=torch.from_numpy(pixels.transpose(2,0,1).copy()).float()[None].to(device)/255
    probabilities=model(x).sigmoid()
    return np.stack([restore_probabilities(probabilities[:,c:c+1],valid,shape).cpu().numpy()
                     for c in range(len(CLASSES))])


def save_lesion_prediction(checkpoint,image_path,output,device='cpu',save_probabilities=False):
    output=Path(output)
    if output.exists():raise FileExistsError('Preserve existing prediction artifacts')
    with Image.open(image_path) as image:probabilities=predict_lesions(checkpoint,image,device)
    output.mkdir(parents=True,exist_ok=False)
    for c,name in enumerate(CLASSES):
        Image.fromarray((probabilities[c]>=.5).astype(np.uint8)*255).save(output/(name+'.png'))
    if save_probabilities:np.save(output/'probabilities.npy',probabilities,allow_pickle=False)
    result={'classes':list(CLASSES),'shape':list(probabilities.shape),'threshold':.5,'device':device,
            'grid':'original image pixels; bilinear probability restoration before threshold',
            'checkpoint_sha256':sha256(checkpoint),'image_sha256':sha256(image_path),
            'probabilities_saved':bool(save_probabilities),'clinical_validation':False}
    write_json(output/'prediction.json',result);return result
