import os
from tqdm import tqdm
from PIL import Image, ImageFile
import cv2
import numpy as np
from hand.models import load_model as load_hand_model
from hand.detect import detect_image as detect_hand
from hand.utils.parse_config import parse_data_config
from hand.utils.utils import load_classes
from fingertip.model import load_model as load_fingertip_model
from fingertip.detect import detect_image as detect_fingertip

ImageFile.LOAD_TRUNCATED_IMAGES = True


IMAGES_INPUT = "D:/Datasets/HaGRID/hagrid-3/images/subsample"
IMAGES_OUTPUT = "output/pipeline"


class DetectEngine:
    HAND_DATA_CONFIG = "config/hagrid-3.data"
    HAND_MODEL_DEF = "config/yolov3-hagrid-3.cfg"
    CONF_THRES = 0.1
    NMS_THRES = 0.4
    HAND_WEIGHTS = "weights/hand/hagrid-3.pth"
    FINGERTIP_WEIGHTS = "weights/fingertip/hagrid-3-fingertip.pth"

    def __init__(self):
        # load models
        self.hand_model = load_hand_model(self.HAND_MODEL_DEF, self.HAND_WEIGHTS)
        self.fingertip_model = load_fingertip_model(self.FINGERTIP_WEIGHTS)

        data_config = parse_data_config(self.HAND_DATA_CONFIG)
        self.class_names = load_classes(data_config["names"])

    def detect(self, image):
        '''Return fingertip position and hand bounding box, confidence and class. All coordinates are absolute values in floating numbers.
        '''
        # hand detection
        detection = detect_hand(self.hand_model, image, conf_thres=self.CONF_THRES, nms_thres=self.NMS_THRES)
        if detection.shape[0] == 0:
            # no hand detected
            return None
        abs_x1, abs_y1, abs_x2, abs_y2, conf, cls_pred = detection[0]
        abs_x1, abs_y1, abs_x2, abs_y2 = max(abs_x1, 0), max(abs_y1, 0), min(abs_x2, image.shape[1]), min(abs_y2, image.shape[0])

        b_x1, b_y1, b_x2, b_y2 = round(abs_x1), round(abs_y1), round(abs_x2), round(abs_y2)
        if b_x1 >= b_x2 or b_y1 >= b_y2:
            return None

        # crop hand
        hand_image = image[b_y1:b_y2, b_x1:b_x2]

        # fingertip detection
        tip_x, tip_y = detect_fingertip(self.fingertip_model, hand_image)
        abs_tip_x, abs_tip_y = tip_x * hand_image.shape[1] + b_x1, tip_y * hand_image.shape[0] + b_y1

        return abs_tip_x, abs_tip_y, abs_x1, abs_y1, abs_x2, abs_y2, conf, cls_pred

    def classIndexToName(self, class_index):
        return self.class_names[int(class_index)]


if __name__ == "__main__":
    backend_engine = DetectEngine()
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

        detection = backend_engine.detect(image)
        if detection is not None:
            x, y, bx1, by1, bx2, by2, conf, cls_ = detection
            x, y, bx1, by1, bx2, by2 = round(x), round(y), round(bx1), round(by1), round(bx2), round(by2)
            # draw fingertip
            image = cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            # draw bounding box
            image = cv2.rectangle(image, (bx1, by1), (bx2, by2), (0, 255, 0), 1)

        # show image
        image = Image.fromarray(image)
        image.save(os.path.join(IMAGES_OUTPUT, file_name))
