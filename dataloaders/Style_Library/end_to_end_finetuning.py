import os
import json
import torch
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from dataloaders.Style_Library import style_dicts
from dataloaders.utils_loader import load_top1_map, load_topk_map


class Train_dataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.flip = cfg.dataset['random_flip']
        self.crop = cfg.dataset['random_crop']

        self.style_dicts = style_dicts
        self.style_ids = list(style_dicts.keys())

        root_dir = os.path.join(cfg.dataset_root, 'Style_transition')

        # Load train splits
        self.name_list = []
        name_list = os.listdir(os.path.join(root_dir, f'01_expertC'))

        datalist_txt = os.path.join(root_dir, f'test.txt')
        with open(datalist_txt, 'r') as fid:
            datanames = fid.readlines()
        testnames = [dataname.strip() for dataname in datanames]

        for img_name in name_list:
            name, ext = os.path.splitext(img_name)
            tag = name.split('-')[0]
            if tag not in testnames:
                self.name_list.append(img_name)

        # Load jsonl file
        train_jsonl = os.path.join(root_dir, 'json/train.jsonl')
        topk_map = load_topk_map(train_jsonl)

        # Load style library
        self.data_dict = {style_idx: [] for style_idx in self.style_dicts}
        for img_name in self.name_list:
            for style_idx, style_name in self.style_dicts.items():
                style_dir = os.path.join(root_dir, f'{style_idx}_{style_name}')
                style_img_path = os.path.join(style_dir, img_name)
                self.data_dict[style_idx].append(style_img_path)

        name_to_index = {name: i for i, name in enumerate(self.name_list)}

        self.ref_cand_indices = {}
        for i, img_name in enumerate(self.name_list):
            cand_names = topk_map.get(img_name)
            cand_idx = [name_to_index[name] for name in cand_names]
            self.ref_cand_indices[i] = cand_idx

        self.transform = transforms.Compose(
            [transforms.Resize(size=(448, 448)),
             transforms.ToTensor()])

        self.n_style_per_batch = cfg.dataset['n_style_per_batch']
        self.n_content_per_batch = cfg.dataset['n_content_per_batch']
        self.n_anchors_per_style = cfg.dataset['n_anchors_per_style']
        self.n_refers_per_anchor = cfg.dataset['n_refers_per_anchor']

        self.items = []
        self.build_epoch()

    def build_epoch(self):
        """
        Construct pairs of (src_style, tgt_style, input image, refer image).
        Ensure each batch contains multiple samples from the same style and different styles,
        so that SupCon loss can be applied effectively.
        """
        seed = random.randint(0, 2 ** 32)
        random.seed(seed)
        np.random.seed(seed)

        N = len(self.name_list)
        S = len(self.style_ids)
        A = self.n_anchors_per_style
        R = self.n_refers_per_anchor
        S_per_batch = self.n_style_per_batch
        C_per_batch = self.n_content_per_batch

        # anchor image sampling
        anchors_per_style = {}
        for style in self.style_ids:
            anchors_per_style[style] = random.sample(range(N), A)

        # Collect n_style * n_style * A * R data
        items = []
        for src_style in self.style_ids:
            anchors = anchors_per_style[src_style]
            for tgt_style in self.style_ids:
                for anchor_idx in anchors:
                    # get reference candidates per anchor image
                    candidates = self.ref_cand_indices[anchor_idx]

                    # reference image sampling
                    refers = random.sample(candidates, R)
                    for refer_idx in refers:
                        items.append((src_style, tgt_style, int(anchor_idx), int(refer_idx)))

        idx_arr = np.arange(len(items)).reshape(S, S, A, R)
        src_style_block = [idx_arr[style_idx].reshape(-1) for style_idx in range(S)]
        for style_block in src_style_block:
            np.random.shuffle(style_block)

        idx_arr = np.array(src_style_block).reshape(S_per_batch, -1, C_per_batch)

        # shuffled along batch
        idx_arr = idx_arr.transpose(1, 0, 2)
        perm = np.random.permutation(len(idx_arr))
        idx_arr = idx_arr[perm]
        idx_arr = idx_arr.flatten()

        items = np.array(items, dtype=object)
        self.items = items[idx_arr].tolist()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        src_style, tgt_style, anchor_idx, refer_idx = self.items[idx]

        # Load paths
        canonicalizer_input_path = self.data_dict[src_style][anchor_idx]  # src style image of input content
        canonicalizer_gt_path = self.data_dict['01'][anchor_idx]  # canonical image of input content
        restyler_refer_path = self.data_dict[tgt_style][refer_idx]  # tgt style image of refer content
        restyler_gt_path = self.data_dict[tgt_style][anchor_idx]  # tgt style image of input content
        canonical_refer_path = self.data_dict['01'][refer_idx]  # canonical image of refer content

        # Load images
        canonicalizer_input = Image.open(canonicalizer_input_path).convert('RGB')
        canonicalizer_gt = Image.open(canonicalizer_gt_path).convert('RGB')
        restyler_refer = Image.open(restyler_refer_path).convert('RGB')
        restyler_gt = Image.open(restyler_gt_path).convert('RGB')
        canonical_refer = Image.open(canonical_refer_path).convert('RGB')

        # Apply same augmentations on same content
        crop1, flip1, params1 = self.get_aug_params(canonicalizer_input)
        canonicalizer_input = self.augment(canonicalizer_input, crop1, flip1, params1)
        canonicalizer_gt = self.augment(canonicalizer_gt, crop1, flip1, params1)
        restyler_gt = self.augment(restyler_gt, crop1, flip1, params1)

        crop2, flip2, params2 = self.get_aug_params(restyler_refer)
        restyler_refer = self.augment(restyler_refer, crop2, flip2, params2)
        canonical_refer = self.augment(canonical_refer, crop2, flip2, params2)

        # To tensor
        canonicalizer_input = self.transform(canonicalizer_input)
        canonicalizer_gt = self.transform(canonicalizer_gt)
        restyler_refer = self.transform(restyler_refer)
        restyler_gt = self.transform(restyler_gt)
        canonical_refer = self.transform(canonical_refer)

        return {'src_style_label': int(src_style),
                'canonicalizer_input': canonicalizer_input,
                'canonicalizer_GT': canonicalizer_gt,
                'tgt_style_label': int(tgt_style),
                'restyler_refer': restyler_refer,
                'restyler_GT': restyler_gt,
                'canonical_refer': canonical_refer}

    def augment(self, img, crop_flag, flip_flag, params):
        if crop_flag and params is not None:
            img = TF.crop(img, *params)
        if flip_flag:
            img = TF.hflip(img)
        return img

    def get_aug_params(self, img, ratio=0.8):
        crop_flag = bool(np.random.randint(0, 2)) if self.crop else False
        flip_flag = bool(np.random.randint(0, 2)) if self.flip else False

        params = None
        if crop_flag:
            w, h = img.size
            ratio_h = np.random.uniform(ratio, 1.0)
            ratio_w = np.random.uniform(ratio, 1.0)

            crop_h = round(h * ratio_h)
            crop_w = round(w * ratio_w)
            params = transforms.RandomCrop.get_params(img, output_size=(crop_h, crop_w))  # i, j, h, w
        return crop_flag, flip_flag, params


class Test_dataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg

        self.style_dicts = style_dicts

        root_dir = os.path.join(cfg.dataset_root, 'Style_transition')

        datalist_txt = os.path.join(root_dir, f'test.txt')
        with open(datalist_txt, 'r') as fid:
            datanames = fid.readlines()
        testnames = [dataname.strip() for dataname in datanames]

        canonical_dir = os.path.join(root_dir, f'01_expertC')
        name_list = []
        for img_name in os.listdir(canonical_dir):
            name, ext = os.path.splitext(img_name)
            tag = name.split('-')[0]
            if tag in testnames:
                name_list.append(img_name)

        random.seed(42)
        n_contents = cfg.testing['n_contents']
        sampled_ids = random.sample(name_list, n_contents)

        test_jsonl = os.path.join(root_dir, 'json/test.jsonl')
        top1_map = load_top1_map(test_jsonl)

        self.pairs = list()
        for src_style_idx, src_style_name in self.style_dicts.items():
            src_img_dir = os.path.join(root_dir, f'{src_style_idx}_{src_style_name}')

            for tgt_style_idx, tgt_style_name in self.style_dicts.items():

                # if src_style_idx == tgt_style_idx:
                #     continue        # Self style transition is not included in test

                if (src_style_idx == '01') or (tgt_style_idx == '01'):
                    continue        # Canonical style is not included in test

                tgt_img_dir = os.path.join(root_dir, f'{tgt_style_idx}_{tgt_style_name}')

                for input_image_name in sampled_ids:
                    refer_image_name = top1_map.get(input_image_name)

                    input_img_path = os.path.join(src_img_dir, input_image_name)
                    refer_img_path = os.path.join(tgt_img_dir, refer_image_name)
                    canonical_gt_path = os.path.join(canonical_dir, input_image_name)
                    restyler_gt_path = os.path.join(tgt_img_dir, input_image_name)
                    canonical_refer_path = os.path.join(canonical_dir, refer_image_name)
                    self.pairs.append((int(src_style_idx), int(tgt_style_idx),
                                       input_img_path, refer_img_path,
                                       canonical_gt_path, restyler_gt_path, canonical_refer_path))

        self.transform = transforms.Compose([transforms.Resize(size=(448, 448)),
                                             transforms.ToTensor()])

    def __getitem__(self, idx):
        src_style, tgt_style, input_img_path, refer_img_path, canonical_gt_path, restyler_gt_path, canonical_refer_path = \
        self.pairs[idx]
        tag = (src_style, tgt_style)

        return {'tag': tag, 'img_name': os.path.basename(input_img_path),
                'canonicalizer_input': self.load_image(input_img_path),
                'canonicalizer_GT': self.load_image(canonical_gt_path),
                'restyler_refer': self.load_image(refer_img_path),
                'restyler_GT': self.load_image(restyler_gt_path),
                'canonical_refer': self.load_image(canonical_refer_path)}

    def __len__(self):
        return len(self.pairs)

    def load_image(self, path):
        img = Image.open(path).convert("RGB")
        return self.transform(img)

