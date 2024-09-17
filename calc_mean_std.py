from pathlib import Path
from collections import defaultdict
import json
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as a
import numpy as np
from skimage import io as skio
from skimage import color
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity
import torch
import random
import os
from split_dataset import seed_all



class COCOParser:
    def __init__(self, anns_file):
        with open(anns_file, 'r') as f:
            coco = json.load(f)

        self.annIm_dict = defaultdict(list)
        self.cat_dict = {}
        self.annId_dict = {}
        self.im_dict = {}
        for ann in coco['annotations']:
            self.annIm_dict[ann['image_id']].append(ann)
            self.annId_dict[ann['id']]=ann
        for img in coco['images']:
            self.im_dict[img['id']] = img
        for cat in coco['categories']:
            self.cat_dict[cat['id']] = cat

    def get_img_ids(self):
        return list(self.im_dict.keys())
    def get_ann_ids(self, im_ids):
        im_ids=im_ids if isinstance(im_ids, list) else [im_ids]
        return [ann['id'] for im_id in im_ids for ann in self.annIm_dict[im_id]]
    def load_anns(self, ann_ids):
        im_ids=ann_ids if isinstance(ann_ids, list) else [ann_ids]
        return [self.annId_dict[ann_id] for ann_id in ann_ids]
    def load_cats(self, class_ids):
        class_ids=class_ids if isinstance(class_ids, list) else [class_ids]
        return [self.cat_dict[class_id] for class_id in class_ids]


def get_mean_std(loader):
    ch_sum, ch_sq_sum, n_batches = 0, 0, 0
    for _, data in loader:
        data = data/255.0
        ch_sum += torch.mean(data, dim=[0,2,3])
        ch_sq_sum += torch.mean(data**2, dim=[0,2,3])
        n_batches +=1
    mean = ch_sum/n_batches
    std = (ch_sq_sum/n_batches-mean**2)**0.5
    return mean, std

def get_files_to_load(train_split, image_folder):
    coco_images = COCOParser(anns_file=train_split)
    file_ls = [coco_images.im_dict[im_path]['file_name'] for im_path in coco_images.im_dict]
    file_ls = list(Path(image_folder).glob("*.png"))
    file_ls = [f"{str(fi.stem)}.png" for fi in file_ls]
    return file_ls

class StomaDataset(Dataset):
    def __init__(self,
                 file_ls,  # list of files to add
                 image_folder,  # path to the image folder
                 transform=None,  # which augmentation transforms to use
                 ):
        self.file_ls = file_ls
        self.transform = transform
        self.images = None
        self.masks = None
        self._load_images(file_ls, image_folder)

    def __len__(self):
        return len(self.images)

    def _load_images(self, file_ls, image_folder):
        # load first image, so we can get the image shape
        img = color.rgb2gray(skio.imread(f"{image_folder}/{file_ls[0]}"))
        images = np.ndarray((len(file_ls), *img.shape, 3), dtype=np.uint8)
        print("Loading Images with 3 Channels, but in gray-scale. Each channel will have the same pixel values.")
        for idx, file in enumerate(file_ls):
            img = color.rgb2gray(skio.imread(f"{image_folder}/{file}"))
            img = img_as_ubyte(img)
            img = rescale_intensity(img)
            images[idx] = np.dstack((img, img, img))
        self.images = images
        return

    def __getitem__(self, idx):
        image = self.images[idx]
        fname = self.file_ls[idx]
        if self.transform:
            augmentations = self.transform(image=image)
            image = augmentations["image"]
        return fname, image

if __name__ == "__main__":
    batch_size = 10
    num_workers = 4
    seed = 42
    seed_all(seed=seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    transforms = a.Compose([ToTensorV2()])
    shuffle = False

    image_folder = "./data/images"
    train_split = f"./data/splits/train.json"

    file_ls = get_files_to_load(train_split, image_folder)
    ds = StomaDataset(file_ls=file_ls, image_folder=image_folder, transform=transforms)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers, generator=generator)

    mean, std = get_mean_std(dataloader)
    print(f"{mean=}")
    print(f"{std=}")
