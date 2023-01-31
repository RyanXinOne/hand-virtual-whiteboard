import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
from model import FingertipDetector
from dataset import Hagrid3IndexFingertipDataset
from utils import device, remove_padding, transform_coordinate_with_padding


PRETRAINED_WEIGHTS = "weights/fingertip/hagrid-3-fingertip.pt"
OUTPUT_DIR = "output/"


os.makedirs(OUTPUT_DIR, exist_ok=True)

sample_dataset = Hagrid3IndexFingertipDataset(dataset='subsample')

model = FingertipDetector().to(device)
if PRETRAINED_WEIGHTS:
    model.load_state_dict(torch.load(PRETRAINED_WEIGHTS))
    print(f"Loaded weights from '{PRETRAINED_WEIGHTS}'")

model.eval()
with torch.no_grad():
    for i, (X, y, abs_pad) in tqdm(enumerate(sample_dataset)):
        pred = model(X.unsqueeze(0)).squeeze(0)

        img = transforms.ToPILImage()(X)
        img = np.array(img)
        # remove padding
        img = remove_padding(img, abs_pad)
        # transform coordinates
        cord_x, cord_y = transform_coordinate_with_padding(y[0].item(), y[1].item(), img.shape[1], img.shape[0], abs_pad)
        pred_x, pred_y = transform_coordinate_with_padding(pred[0].item(), pred[1].item(), img.shape[1], img.shape[0], abs_pad)
        # calculate absolute coordinates
        abs_cord_x, abs_cord_y = int(cord_x * img.shape[1]), int(cord_y * img.shape[0])
        abs_pred_x, abs_pred_y = int(pred_x * img.shape[1]), int(pred_y * img.shape[0])
        img = cv2.circle(img, (abs_cord_x, abs_cord_y), 3, (255, 0, 0), -1)
        img = cv2.circle(img, (abs_pred_x, abs_pred_y), 3, (0, 0, 255), -1)
        img = Image.fromarray(img)
        img.save(os.path.join(OUTPUT_DIR, f"sample{i}.jpg"))
