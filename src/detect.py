import os
from tqdm import tqdm
from PIL import Image, ImageFile
import cv2
import numpy as np
import torch

from hand.models import load_model as load_hand_model
from hand.detect import detect_image as detect_hand
from hand.utils.parse_config import parse_data_config
from hand.utils.utils import load_classes
from fingertip.model import load_model as load_fingertip_model
from fingertip.detect import detect_image as detect_fingertip

ImageFile.LOAD_TRUNCATED_IMAGES = True


IMAGES_INPUT = "D:/Datasets/HaGRID/hagrid-13/images/subsample"
IMAGES_OUTPUT = "output/pipeline"


class DetectEngine:
    MAX_NUM = 2
    HAND_DATA_CONFIG = "config/hagrid-13.data"
    HAND_MODEL_DEF = "config/yolov3-hagrid-13.cfg"
    HAND_WEIGHTS = "weights/hand/hagrid-13.pth"
    CONF_THRES = 0.25
    NMS_THRES = 0.4
    FINGERTIP_WEIGHTS = "weights/fingertip/hagrid-13-fingertip.pth"
    FINGERTIP_CLASSES = ("like", "one", "stop", "two_up")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        # load models
        self.hand_model = load_hand_model(self.HAND_MODEL_DEF, self.HAND_WEIGHTS, device=self.DEVICE)
        self.fingertip_model = load_fingertip_model(self.FINGERTIP_WEIGHTS, device=self.DEVICE)

        data_config = parse_data_config(self.HAND_DATA_CONFIG)
        self.class_names = load_classes(data_config["names"])

    def detect(self, image):
        '''Return a list of detections. Each detection is a tuple containing fingertip coordinate, hand bounding box, confidence, and class name.

        Detections are returned in the order of confidence.

        All coordinates are absolute values in floating numbers.

        If no fingertip is detected, fingertip coordinate is set to -1.
        '''
        # hand detection
        hand_detections = detect_hand(self.hand_model, image, conf_thres=self.CONF_THRES, nms_thres=self.NMS_THRES, device=self.DEVICE)

        detections = []
        for hand_detection in hand_detections[:self.MAX_NUM]:
            abs_x1, abs_y1, abs_x2, abs_y2, conf, cls_pred = hand_detection
            abs_x1, abs_y1, abs_x2, abs_y2 = max(abs_x1, 0), max(abs_y1, 0), min(abs_x2, image.shape[1]), min(abs_y2, image.shape[0])

            b_x1, b_y1, b_x2, b_y2 = round(abs_x1), round(abs_y1), round(abs_x2), round(abs_y2)
            if b_x1 >= b_x2 or b_y1 >= b_y2:
                continue

            cls_name = self.classIndexToName(cls_pred)
            if cls_name in self.FINGERTIP_CLASSES:
                # crop hand
                hand_image = image[b_y1:b_y2, b_x1:b_x2]

                # fingertip detection
                tip_x, tip_y = detect_fingertip(self.fingertip_model, hand_image, device=self.DEVICE)
                abs_tip_x, abs_tip_y = tip_x * hand_image.shape[1] + b_x1, tip_y * hand_image.shape[0] + b_y1
            else:
                abs_tip_x = abs_tip_y = -1

            detections.append((abs_tip_x, abs_tip_y, abs_x1, abs_y1, abs_x2, abs_y2, conf, cls_name))
        return detections

    def classIndexToName(self, class_index):
        return self.class_names[int(class_index)]

    def drawDetection(self, image, detection):
        '''Draw detection on image in RGB format.
        '''
        x, y, bx1, by1, bx2, by2, conf, cls_n = detection
        x, y, bx1, by1, bx2, by2 = round(x), round(y), round(bx1), round(by1), round(bx2), round(by2)
        if x > -1:
            # draw fingertip
            image = cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
        # draw bounding box
        image = cv2.rectangle(image, (bx1, by1), (bx2, by2), (0, 0, 255), 1)
        # draw class name
        image = cv2.putText(image, f"{cls_n} {conf:.2f}", (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return image


if __name__ == "__main__":
    engine = DetectEngine()
    # load images
    image_files = os.listdir(IMAGES_INPUT)
    os.makedirs(IMAGES_OUTPUT, exist_ok=True)
    for file_name in tqdm(image_files):
        # read image
        try:
            img_path = os.path.join(IMAGES_INPUT, file_name)
            image = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except:
            print(f"Could not read image '{img_path}'.")
            continue

        detections = engine.detect(image)
        for detection in detections:
            image = engine.drawDetection(image, detection)

        # show image
        image = Image.fromarray(image)
        image.save(os.path.join(IMAGES_OUTPUT, file_name))
