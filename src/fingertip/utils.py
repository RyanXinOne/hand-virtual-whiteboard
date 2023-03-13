import imgaug.augmenters as iaa
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms


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
        target_size = (size, max(round(size * image.shape[2] / image.shape[1]), 1))
    else:
        target_size = (max(round(size * image.shape[1] / image.shape[2]), 1), size)
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

    abs_x, abs_y, abs_w, abs_h = round(x * image.shape[1]), round(y * image.shape[0]), round(w * image.shape[1]), round(h * image.shape[0])
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


augmentation = iaa.Sequential([
    iaa.Dropout([0.0, 0.01]),
    iaa.Sharpen((0.0, 0.1)),
    iaa.Affine(rotate=(-20, 20), translate_percent=(-0.05, 0.05), scale=(0.8, 1.1)),
    iaa.AddToBrightness((-60, 40)),
    iaa.AddToHue((-20, 20)),
    iaa.Fliplr(0.5),
])


def augment_image(image, keypoint):
    '''Augment an image.

    It is guaranteed that the keypoint is still in the image after augmentation.

    Args:
        image (Tensor): Image to be augmented.
        keypoint (tuple): Keypoint of relative coordinates of the image.

    Returns:
        Tensor: Augmented image.
        tuple: Relevant keypoint.
    '''
    image = np.array(transforms.ToPILImage()(image))
    width, height = image.shape[1], image.shape[0]

    n_keypoint = (-1, -1)
    while not (n_keypoint[0] >= 0 and n_keypoint[0] < width and n_keypoint[1] >= 0 and n_keypoint[1] < height):
        n_image, n_keypoint = augmentation(image=image, keypoints=(keypoint[0] * width, keypoint[1] * height))

    return transforms.ToTensor()(n_image), (n_keypoint[0] / width, n_keypoint[1] / height)
