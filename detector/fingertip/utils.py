import numpy as np
import torch
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pad_to_square_image(image, pad_value=0):
    '''Pad to square image.

    Args:
        image (Tensor): Image to be padded.
        pad_value (int): Value to be used for padding.

    Returns:
        Tensor: Padded image.
    '''
    c, h, w = image.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    abs_pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    image = F.pad(image, abs_pad, "constant", value=pad_value)

    return image, abs_pad


def remove_padding(image, abs_pad):
    '''Remove padding from an image.

    Args:
        image (Array): Image to be padded.
        abs_pad (tuple): Absolute padding (l, r, t, b) of the image.

    Returns:
        Array: Unpadded image.
    '''
    l, r, t, b = abs_pad
    image = image[t:image.shape[0]-b, l:image.shape[1]-r]
    return image


def resize_image(image, size):
    '''Resize a image by proportional scaling.

    Args:
        image (Tensor): Image to be resized.
        size (int): Length of the longest side of resized image.

    Returns:
        Tensor: Resized image.
    '''
    if image.shape[1] > image.shape[2]:
        target_size = (size, max(int(size * image.shape[2] / image.shape[1]), 1))
    else:
        target_size = (max(int(size * image.shape[1] / image.shape[2]), 1), size)
    image = F.interpolate(image.unsqueeze(0), size=target_size, mode="nearest").squeeze(0)
    return image


def crop_image(image, x, y, w, h, anchor='top-left'):
    '''Crop an image.

    Args:
        image (Array): Image to be cropped.
        x, y, w, h (float): Relative bounding box [0, 1] of the crop region.
        anchor (str): Anchor of the bounding box. Options: 'top-left', 'center'.

    Returns:
        Array: Cropped image.
    '''
    if anchor == 'center':
        x -= w / 2
        y -= h / 2

    abs_x, abs_y, abs_w, abs_h = int(x * image.shape[1]), int(y * image.shape[0]), int(w * image.shape[1]), int(h * image.shape[0])
    image = image[abs_y:abs_y+abs_h, abs_x:abs_x+abs_w]
    return image


def transform_coordinate_without_padding(in_x, in_y, img_width, img_height, abs_pad):
    '''Transform relative coordinates of an image into one after adding padding.

    Args:
        in_x, in_y (float): Relative coordinates [0, 1] to be transformed.
        img_width (int): Width of the padded image.
        img_height (int): Height of the padded image.
        abs_pad (tuple): Absolute padding (l, r, t, b) of the image.

    Returns:
        out_x, out_y (float): Transformed relative coordinates.
    '''
    out_x = (in_x * (img_width - abs_pad[0] - abs_pad[1]) + abs_pad[0]) / img_width
    out_y = (in_y * (img_height - abs_pad[2] - abs_pad[3]) + abs_pad[2]) / img_height
    return out_x, out_y


def transform_coordinate_with_padding(in_x, in_y, img_width, img_height, abs_pad):
    '''Transform relative coordinates of an image into one after removing padding.

    Args:
        in_x, in_y (float): Relative coordinates [0, 1] to be transformed.
        img_width (int): Width of the padded image.
        img_height (int): Height of the padded image.
        abs_pad (tuple): Absolute padding (l, r, t, b) of the image.

    Returns:
        out_x, out_y (float): Transformed relative coordinates.
    '''
    out_x = (in_x * img_width - abs_pad[0]) / (img_width - abs_pad[0] - abs_pad[1])
    out_y = (in_y * img_height - abs_pad[2]) / (img_height - abs_pad[2] - abs_pad[3])
    return out_x, out_y
