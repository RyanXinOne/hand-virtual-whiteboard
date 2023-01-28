import torch
import torch.nn.functional as F
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pad_to_square(img, pad_value=0):
    '''Pad to square image.

    Args:
        img (Tensor): Image to be padded.
        pad_value (int): Value to be used for padding.
    '''
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    '''Resize a square image.
    '''
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image
