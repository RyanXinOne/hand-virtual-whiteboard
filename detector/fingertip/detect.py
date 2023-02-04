import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from model import load_model
from dataset import Hagrid3IndexFingertipDataset
from utils import resize_image, pad_to_square_image, transform_coordinate_with_padding


PRETRAINED_WEIGHTS = "weights/fingertip/hagrid-3-fingertip.pth"
OUTPUT_DIR = "output/fingertip"


def detect_image(model, image, image_size=128):
    '''Detect index fingertip in a single image.

    Args:
        model (nn.Module): model to use for detection
        image (tensor): image to detect
        image_size (int): size of the image to be fed into the model

    Returns:
        (tuple): predicted relative coordinates of the index fingertip
    '''
    # transform image
    image = resize_image(image, image_size)
    image, abs_pad = pad_to_square_image(image)
    # model evaluation
    model.eval()
    with torch.no_grad():
        pred = model(image.unsqueeze(0)).squeeze(0)
    # transform coordinates back
    pred_x, pred_y = transform_coordinate_with_padding(pred[0].item(), pred[1].item(), image.shape[2], image.shape[1], abs_pad)
    return (pred_x, pred_y)


def draw_and_save_output_image(image, detection, output_path, label=None):
    '''Draw predicted and ground truth coordinates on the image and save it.

    Args:
        image (np.array): image to draw on
        detection (tuple): predicted relative coordinates of the index fingertip
        output_path (str): path to save the image
        label (tuple): ground truth relative coordinates of the index fingertip
    '''
    # draw marker
    if label is not None:
        abs_cord_x, abs_cord_y = int(label[0] * image.shape[1]), int(label[1] * image.shape[0])
        image = cv2.circle(image, (abs_cord_x, abs_cord_y), 3, (255, 0, 0), -1)
    # calculate absolute coordinates
    abs_pred_x, abs_pred_y = int(detection[0] * image.shape[1]), int(detection[1] * image.shape[0])
    image = cv2.circle(image, (abs_pred_x, abs_pred_y), 3, (0, 0, 255), -1)
    # save image
    image = Image.fromarray(image)
    image.save(output_path)


if __name__ == "__main__":
    sample_dataset = Hagrid3IndexFingertipDataset(dataset='subsample', learning=False)
    model = load_model(weights_path=PRETRAINED_WEIGHTS)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i, sample in tqdm(enumerate(sample_dataset)):
        if sample is None:
            continue
        X, y, img_name = sample
        pred = detect_image(model, X)
        X, y = np.array(transforms.ToPILImage()(X)), y.tolist()
        draw_and_save_output_image(X, pred, os.path.join(OUTPUT_DIR, f"{img_name}.jpg"), label=y)
