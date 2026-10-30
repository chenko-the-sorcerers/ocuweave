"""Binary lesion inference at the input image geometry; no clinical diagnosis."""
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from ocuweave.models.factory import make_model
from ocuweave.data.preprocessing.geometry import letterbox


@torch.no_grad()
def predict_image(checkpoint, image, device='cpu'):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = make_model(state['config']).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    image = image.convert('RGB')
    size = state['config']['size']
    pixels, _, _ = letterbox(image, np.zeros((image.height,image.width),bool), size)
    tensor = torch.from_numpy(pixels.transpose(2,0,1).copy()).float().unsqueeze(0).to(device) / 255
    probability = model(tensor).sigmoid()
    scale = min(size/image.width,size/image.height)
    width, height = max(1,round(image.width*scale)), max(1,round(image.height*scale))
    left, top = (size-width)//2, (size-height)//2
    cropped = probability[:,:,top:top+height,left:left+width]
    restored = F.interpolate(cropped,size=(image.height,image.width),mode='bilinear',align_corners=False)
    return restored[0,0].cpu().numpy()
