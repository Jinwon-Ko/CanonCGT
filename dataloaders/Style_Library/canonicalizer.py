import os
import torch
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

from dataloaders.Style_Library import style_dicts


class Train_dataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.flip = cfg.dataset['random_flip']
        self.crop = cfg.dataset['random_crop']
        self.rotate = cfg.dataset['random_rotate']

        self.style_dicts = style_dicts

        root_dir = os.path.join(cfg.dataset_root, 'Style_transition')

        datalist_txt = os.path.join(root_dir, f'test.txt')
        with open(datalist_txt, 'r') as fid:
            datanames = fid.readlines()
        tag_names = [dataname.strip() for dataname in datanames]

        self.samples = list()
        canonical_dir = os.path.join(root_dir, f'01_expertC')
        for style_idx, style_name in self.style_dicts.items():
            img_dir = os.path.join(root_dir, f'{style_idx}_{style_name}')
            datalist = os.listdir(img_dir)
            datalist = sorted(datalist)

            for img_name in datalist:
                name, ext = os.path.splitext(img_name)
                tag = name.split('-')[0]
                if tag not in tag_names:
                    img_path = os.path.join(img_dir, img_name)
                    canonical_path = os.path.join(canonical_dir, img_name)
                    self.samples.append((img_path, canonical_path, str(style_idx)))

        self.transform = transforms.Compose(
            [transforms.Resize(size=(448, 448)),
             transforms.ToTensor()])

    def __getitem__(self, idx):
        seed = random.randint(0, 2 ** 32)
        random.seed(seed)
        np.random.seed(seed)

        crop = np.random.randint(0, 2) if self.crop else 0
        flip = np.random.randint(0, 2) if self.flip else 0
        rotate = np.random.randint(0, 4) if self.rotate else 0

        img_path, canonical_path, style_label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        gt = Image.open(canonical_path).convert("RGB")

        img, gt = self.augment_pair(img, gt, crop, flip, rotate)

        batch = {'img': self.transform(img),
                 'gt': self.transform(gt),
                 'style_idx': style_label,
                 'img_name': os.path.basename(canonical_path)}

        return batch

    def augment_pair(self, img, gt, crop=0, flip=0, rotate=0):

        if crop == 1:
            i, j, h, w = self.get_crop_params(img, ratio=0.8)
            img = TF.crop(img, i, j, h, w)
            gt = TF.crop(gt, i, j, h, w)

        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)

        if rotate != 0:
            img = img.rotate(90 * rotate, expand=1)
            gt = gt.rotate(90 * rotate, expand=1)

        return img, gt

    def get_crop_params(self, img, ratio=0.8):
        w, h = img.size
        ratio_h = np.random.uniform(ratio, 1.0)
        ratio_w = np.random.uniform(ratio, 1.0)

        crop_h = round(h * ratio_h)
        crop_w = round(w * ratio_w)
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(crop_h, crop_w))
        return i, j, h, w

    def __len__(self):
        return len(self.samples)


class Test_dataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg

        self.style_dicts = style_dicts

        root_dir = os.path.join(cfg.dataset_root, 'Style_transition')

        datalist_txt = os.path.join(root_dir, f'test.txt')
        with open(datalist_txt, 'r') as fid:
            datanames = fid.readlines()
        tag_names = [dataname.strip() for dataname in datanames]

        self.samples = list()
        canonical_dir = os.path.join(root_dir, f'01_expertC')
        datalist = os.listdir(canonical_dir)
        for img_name in datalist:
            name, ext = os.path.splitext(img_name)
            tag = name.split('-')[0]
            if tag in tag_names:
                canonical_path = os.path.join(canonical_dir, img_name)

                style_paths = []
                for style_idx, style_name in self.style_dicts.items():
                    img_dir = os.path.join(root_dir, f'{style_idx}_{style_name}')
                    img_path = os.path.join(img_dir, img_name)
                    style_paths.append((img_path, str(style_idx)))

                self.samples.append((style_paths, canonical_path))

        self.transform = transforms.ToTensor()

    def __getitem__(self, idx):
        style_paths, canonical_path = self.samples[idx]

        img_list = [self.transform(Image.open(img_path).convert("RGB")) for img_path, _ in style_paths]
        img_tensor = torch.stack(img_list, dim=0)
        gt = self.transform(Image.open(canonical_path).convert("RGB"))
        style_list = [style_idx for _, style_idx in style_paths]

        batch = {'img': img_tensor,
                 'gt': gt,
                 'style_idx': style_list,
                 'img_name': os.path.basename(canonical_path)}

        return batch

    def __len__(self):
        return len(self.samples)






