"""Binary lesion inference at the input image geometry; no clinical diagnosis."""
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from ocuweave.models.factory import make_model
from ocuweave.data.preprocessing.geometry import letterbox


@torch.no_grad()
def predict_image(checkpoint, image, device='cpu'):
    from ocuweave.training.engine import select_device
    from ocuweave.evaluation.original_grid import restore_probabilities
    device = select_device(device)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = make_model(state['config']).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    image = image.convert('RGB')
    size = state['config']['size']
    pixels, _, valid = letterbox(image, np.zeros((image.height,image.width),bool), size)
    tensor = torch.from_numpy(pixels.transpose(2,0,1).copy()).float().unsqueeze(0).to(device) / 255
    probability = model(tensor).sigmoid()
    restored = restore_probabilities(probability, valid, (image.height, image.width))
    return restored.cpu().numpy()


def save_prediction(checkpoint, image_path, output, *, device='cpu'):
    """Persist an original-grid prediction without overwriting previous artifacts."""
    from ocuweave import __version__
    from ocuweave.utils.provenance import sha256, write_json
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new prediction directory')
    with Image.open(image_path) as image:
        probability = predict_image(checkpoint, image, device)
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / 'probability.npy', probability)
    Image.fromarray((probability >= 0.5).astype(np.uint8) * 255).save(output / 'mask.png')
    result = {'software_version': __version__, 'checkpoint_sha256': sha256(checkpoint),
              'input_sha256': sha256(image_path), 'threshold': 0.5,
              'output_shape': list(probability.shape), 'clinical_use': False,
              'interpretation': 'Unvalidated binary foreground probability; not disease activity or prognosis.'}
    write_json(output / 'prediction.json', result)
    return result
