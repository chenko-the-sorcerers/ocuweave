"""Predict an exploratory binary lesion mask using a local OcuWeave checkpoint."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from ocuweave.inference.segmentation import predict_image
from ocuweave.utils.provenance import write_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','image','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cpu')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():raise FileExistsError('Choose a new prediction directory')
    with Image.open(args.image) as image:
        probability = predict_image(args.checkpoint,image,args.device)
    output.mkdir(parents=True)
    np.save(output/'probability.npy',probability)
    Image.fromarray((probability>=.5).astype(np.uint8)*255).save(output/'mask.png')
    write_json(output/'prediction.json',{
        'checkpoint_sha256':sha256(args.checkpoint),'input_sha256':sha256(args.image),
        'threshold':.5,'output_shape':list(probability.shape),
        'clinical_use':False,'interpretation':'Unvalidated binary lesion probability; not disease activity or prognosis.'})


if __name__ == '__main__':main()
