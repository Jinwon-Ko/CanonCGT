import os
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

from dataloaders.Style_Library import style_dicts


class Train_dataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg

        self.style_dicts = style_dicts

        root_dir = os.path.join(cfg.dataset_root, 'Style_transition')

        datalist_txt = os.path.join(root_dir, f'test.txt')
        with open(datalist_txt, 'r') as fid:
            datanames = fid.readlines()
        tag_names = [dataname.strip() for dataname in datanames]

        self.samples = list()
        for style_idx, style_name in self.style_dicts.items():
            img_dir = os.path.join(root_dir, f'{style_idx}_{style_name}')
            datalist = os.listdir(img_dir)
            datalist = sorted(datalist)

            for img_name in datalist:
                name, ext = os.path.splitext(img_name)
                tag = name.split('-')[0]
                if tag not in tag_names:
                    img_path = os.path.join(img_dir, img_name)
                    self.samples.append((img_path, int(style_idx)))

        self.transform1 = transforms.Compose(
            [transforms.Resize(size=(448, 448)),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor()])

        self.transform2 = transforms.Compose(
            [transforms.RandomResizedCrop(size=448, scale=(0.25, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor()])

    def __getitem__(self, idx):
        path, style_label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img1 = self.transform1(img)
        img2 = self.transform2(img)

        imgs = torch.stack([img1, img2], dim=0)
        labels = torch.tensor([style_label, style_label], dtype=torch.long)
        return imgs, labels

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
        for style_idx, style_name in self.style_dicts.items():
            img_dir = os.path.join(root_dir, f'{style_idx}_{style_name}')
            datalist = os.listdir(img_dir)
            datalist = sorted(datalist)

            for img_name in datalist:
                name, ext = os.path.splitext(img_name)
                tag = name.split('-')[0]
                if tag in tag_names:
                    img_path = os.path.join(img_dir, img_name)
                    self.samples.append((img_path, int(style_idx)))

        self.transform = transforms.Compose(
            [transforms.Resize(size=(448, 448)),
             transforms.ToTensor()])

    def __getitem__(self, idx):
        path, style_label = self.samples[idx]
        img = self.transform(Image.open(path).convert("RGB"))
        return img, style_label

    def __len__(self):
        return len(self.samples)






