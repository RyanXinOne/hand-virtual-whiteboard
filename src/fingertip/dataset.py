import os
import json
from PIL import Image
from PIL import ImageFile
import numpy as np
import torch
from torch.utils.data import Dataset, default_collate
import torchvision.transforms as transforms

from fingertip.utils import crop_image, resize_image, pad_to_square_image, transform_coordinate_without_padding, augment_image

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Hagrid3IndexFingertipDataset(Dataset):
    '''Hagrid 3-class fingertip dataset.
    '''
    ANNOTATIONS_DIR = 'D:/Datasets/HaGRID/hagrid-13/annos'
    IMAGES_DIR = 'D:/Datasets/HaGRID/hagrid-13/images'
    IMAGE_SIZE = 128

    def __init__(self, dataset='train', learning=True):
        '''
        Args:
            dataset (str): 'train', 'test', 'subsample'
            learning (bool): if True, pre-transformation is applied for learning, including resizing and padding
        '''
        if dataset not in ('train', 'test', 'subsample'):
            raise ValueError(f"Invalid dataset '{dataset}'.")
        self.dataset = dataset
        self.img_names = os.listdir(os.path.join(self.ANNOTATIONS_DIR, self.dataset))
        self.img_names = [os.path.splitext(img_name)[0] for img_name in self.img_names]
        self.learning = learning

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]

        # read image
        try:
            img_path = os.path.join(self.IMAGES_DIR, self.dataset, img_name + '.jpg')
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except:
            print(f"Could not read image '{img_path}'.")
            return

        # read annotation
        try:
            anno_path = os.path.join(self.ANNOTATIONS_DIR, self.dataset, img_name + '.json')
            anno = json.load(open(anno_path, encoding='utf-8'))
        except:
            print(f"Could not read annotation '{anno_path}'.")
            return
        lx, ly, w, h = anno['bboxes'][0]
        o_tip_x, o_tip_y = anno['index_finger_tips'][0]

        # crop image
        img = crop_image(img, lx, ly, w, h)

        # prepare data, label
        img = transforms.ToTensor()(img)
        tip_x, tip_y = (o_tip_x - lx) / w, (o_tip_y - ly) / h
        if not (tip_x >= 0 and tip_x < 1 and tip_y >= 0 and tip_y < 1):
            # print(f"Invalid fingertip coordinate of image '{img_name}'.")
            return

        if self.learning:
            # transform image
            img = resize_image(img, self.IMAGE_SIZE)
            img, abs_pad = pad_to_square_image(img)

            # transform fingertip coordinate
            tip_x, tip_y = transform_coordinate_without_padding(tip_x, tip_y, img.shape[2], img.shape[1], abs_pad)

            # augment image
            img, (tip_x, tip_y) = augment_image(img, (tip_x, tip_y))

        coords = torch.tensor([tip_x, tip_y], dtype=torch.float32)

        return img, coords, img_name

    @classmethod
    def collate_fn(cls, batch):
        # drop invalid samples
        batch = [(sample[0], sample[1]) for sample in batch if sample is not None]

        return default_collate(batch)


if __name__ == '__main__':
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import cv2
    from PIL import Image
    dataset = Hagrid3IndexFingertipDataset(dataset='subsample', learning=True)
    dl = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=Hagrid3IndexFingertipDataset.collate_fn)
    for imgs, cords in tqdm(dl):
        img, cord = imgs[0], cords[0]
        img = transforms.ToPILImage()(img)
        img = np.array(img)
        abs_cord_x, abs_cord_y = int(cord[0].item() * img.shape[1]), int(cord[1].item() * img.shape[0])
        cv2.circle(img, (abs_cord_x, abs_cord_y), 3, (255, 0, 0), -1)
        img = Image.fromarray(img)
        img.show()
        break
