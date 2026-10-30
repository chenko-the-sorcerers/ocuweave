"""Aspect-preserving letterbox with nearest-neighbour target interpolation."""
import numpy as np
from PIL import Image


def letterbox(image, target, size):
    if size < 16 or size % 16:
        raise ValueError('Image size must be a positive multiple of 16')
    scale=min(size/image.width,size/image.height)
    w,h=max(1,round(image.width*scale)),max(1,round(image.height*scale))
    x,y=(size-w)//2,(size-h)//2
    canvas=Image.new('RGB',(size,size))
    canvas.paste(image.resize((w,h),Image.Resampling.BILINEAR),(x,y))
    mask=Image.new('L',(size,size))
    mask.paste(Image.fromarray(target.astype(np.uint8)*255).resize((w,h),Image.Resampling.NEAREST),(x,y))
    valid=np.zeros((size,size),dtype=np.float32);valid[y:y+h,x:x+w]=1
    return np.asarray(canvas).copy(),(np.asarray(mask)>0).astype(np.float32),valid
