import os
import json
import torch
from torch.utils.data import Dataset, default_collate
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from PIL import ImageFile
from utils import device, pad_to_square_image, resize_image, crop_image, transform_coordinate_without_padding

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Hagrid3IndexFingertipDataset(Dataset):
    '''Hagrid 3-class fingertip dataset.
    '''
    ANNOTATIONS_DIR = 'D:/Datasets/HaGRID/hagrid-3/annos'
    IMAGES_DIR = 'D:/Datasets/HaGRID/hagrid-3/images'
    IMAGE_SIZE = 128

    def __init__(self, dataset='train'):
        if dataset not in ('train', 'test', 'subsample'):
            raise ValueError(f"Invalid dataset '{dataset}'.")
        self.dataset = dataset
        self.img_names = os.listdir(os.path.join(self.ANNOTATIONS_DIR, self.dataset))
        self.img_names = [os.path.splitext(img_name)[0] for img_name in self.img_names]

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

        # transform image
        img = transforms.ToTensor()(img)
        img, abs_pad = pad_to_square_image(img)
        img = resize_image(img, self.IMAGE_SIZE)

        # transform fingertip coordinate
        tip_x, tip_y = (o_tip_x - lx) / w, (o_tip_y - ly) / h
        if not (tip_x >= 0 and tip_x < 1 and tip_y >= 0 and tip_y < 1):
            # print(f"Invalid fingertip coordinate of image '{img_name}'.")
            return
        tip_x, tip_y = transform_coordinate_without_padding(tip_x, tip_y, img.shape[2], img.shape[1], abs_pad)
        coords = torch.tensor([tip_x, tip_y], dtype=torch.float32)

        return img.to(device), coords.to(device), abs_pad

    @classmethod
    def collate_fn(cls, batch):
        # drop invalid samples
        batch = [(sample[0], sample[1]) for sample in batch if sample is not None]

        return default_collate(batch)


if __name__ == '__main__':
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    dataset = Hagrid3IndexFingertipDataset()
    dl = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=Hagrid3IndexFingertipDataset.collate_fn)
    for img, coords in tqdm(dl):
        pass
