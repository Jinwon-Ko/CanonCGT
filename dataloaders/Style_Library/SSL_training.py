import os
import math
import torch
import random
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler

from dataloaders.Style_Library import style_dicts
from dataloaders.utils_loader import (load_top1_map, load_topk_map,
                                      _in_center_E, _box_iou_xyxy,
                                      get_style_augment_params,
                                      apply_style_augment_with_params)


class BatchSampler(Sampler):
    def __init__(self, kk_indices, uu_indices, kk_bs, uu_bs, drop_last=True):
        assert (kk_bs > 0) and (uu_bs > 0)

        self.kk_indices = np.array(kk_indices, dtype=np.int64)
        self.uu_indices = np.array(uu_indices, dtype=np.int64)
        self.kk_bs = int(kk_bs)
        self.uu_bs = int(uu_bs)
        self.drop_last = drop_last

    def __len__(self):
        if self.drop_last:
            return len(self.kk_indices) // self.kk_bs
        else:
            return math.ceil(len(self.kk_indices) / self.kk_bs)

    def __iter__(self):
        kk_perm = self.kk_indices.copy()
        np.random.shuffle(kk_perm)
        start_kk = 0

        uu_perm = self.uu_indices.copy()
        np.random.shuffle(uu_perm)
        start_uu = 0

        while start_kk < len(kk_perm):
            end_kk = start_kk + self.kk_bs
            if end_kk > len(kk_perm):
                if self.drop_last:
                    break
                else:
                    batch_kk = kk_perm[start_kk:len(kk_perm)]
                    batch_uu = []
                    yield  list(batch_kk) + list(batch_uu)
                    break
            else:
                batch_kk = kk_perm[start_kk:end_kk]
                start_kk = end_kk

            end_uu = start_uu + self.uu_bs
            if end_uu > len(uu_perm):
                # Take the rest first
                first = uu_perm[start_uu:]

                # Refill to the required count
                uu_perm = self.uu_indices.copy()
                np.random.shuffle(uu_perm)
                need = self.uu_bs - len(first)
                second = uu_perm[:need]

                batch_uu = np.concatenate([first, second], axis=0)
                start_uu = need
            else:
                batch_uu = uu_perm[start_uu:end_uu]
                start_uu = end_uu

            yield batch_kk.tolist() + batch_uu.tolist()


def make_mixed_train_loader(cfg, kk_train_dataset, uu_train_dataset):

    kk_batch_size = cfg.dataset['kk_batch_size']
    uu_batch_size = cfg.dataset['uu_batch_size']

    mixed_dataset = ConcatDataset([kk_train_dataset, uu_train_dataset])

    kk_indices = list(range(0, len(kk_train_dataset)))
    uu_indices = list(range(len(kk_train_dataset), len(kk_train_dataset) + len(uu_train_dataset)))

    batch_sampler = BatchSampler(kk_indices, uu_indices, kk_batch_size, uu_batch_size, drop_last=True)
    loader = DataLoader(mixed_dataset, batch_sampler=batch_sampler, num_workers=cfg.dataset['num_workers'], shuffle=False)
    return loader


# ===================================================================================================
# For Unknown Style dataset [DF2K (DIV2K & Flickr2K) + LSDIR]
# DIV2K     : total number of data : 900  --> train & test split    :    800 & 100
# Flickr2K  : total number of data : 2650 --> train & test split    :  2,650 & 0
# LSDIR     : total number of data : 84,991 --> train & test split  : 72,000 & 8,000 (except the last zip (shard-16))

class Unseen_style_train_dataset(Dataset):
    def __init__(self, cfg, max_tries=10):

        self.max_tries = max_tries

        self.paths = []
        flickr2k_dir = os.path.join(cfg.dataset_root, 'Super_Resolution/Flickr2K/Flickr2K_HR')
        flickr2k_list = sorted(os.listdir(flickr2k_dir))
        self.paths.extend([os.path.join(flickr2k_dir, img_name) for img_name in flickr2k_list])

        lsdir_root = os.path.join(cfg.dataset_root, 'Restoration/LSDIR')
        for lsdir_dir in os.listdir(lsdir_root):
            lsdir_list = sorted(os.listdir(os.path.join(lsdir_root, lsdir_dir)))
            lsdir_list = lsdir_list[:int(len(lsdir_list) * 0.9)]
            self.paths.extend([os.path.join(lsdir_root, lsdir_dir, img_name) for img_name in lsdir_list])

        ppr10k_dir = os.path.join(cfg.dataset_root, 'Retouching/PPR10K/train/target_a_jpg')
        ppr10k_list = sorted(os.listdir(ppr10k_dir))
        self.paths.extend([os.path.join(ppr10k_dir, img_name) for img_name in ppr10k_list])

        self.to_tensor = transforms.ToTensor()
        self.random_hflip = transforms.RandomHorizontalFlip(p=0.5)

    def __len__(self):
        return len(self.paths)

    def _to_tensor(self, pil_img):
        return self.to_tensor(pil_img)

    @torch.no_grad()
    def __getitem__(self, idx):

        seed = random.randint(0, 2 ** 32)
        random.seed(seed)
        np.random.seed(seed)

        path = self.paths[idx]
        src = Image.open(path).convert("RGB")

        view_A, view_B = self.get_two_crops_with_IoU_constraint(src, max_iou=0.5)
        view_inp, view_ref, view_tgt = self.generate_SSL_pair(view_A, view_B)

        inp = self._to_tensor(view_inp)
        ref = self._to_tensor(view_ref)
        tgt = self._to_tensor(view_tgt)

        return {'canonicalizer_input': inp,                  # X
                'canonicalizer_GT': torch.zeros_like(tgt),   # Y_gt (= Y_0)
                'restyler_refer': ref,                       # R
                'restyler_GT': tgt,                          # Z_gt
                'canonical_refer': torch.zeros_like(ref),    # R_0
                'pair_type': 'uu'}

    def get_two_crops_with_IoU_constraint(self, pil_img,
                                          size=(448, 448), scale=(0.1, 0.6), ratio=(3/4, 4/3),
                                          max_iou=0.5, box1_max_tries=10, box2_max_tries=10, a=0.25):

        W, H = pil_img.size

        # Generate Box 1
        for _ in range(box1_max_tries):
            i1, j1, h1, w1 = transforms.RandomResizedCrop.get_params(pil_img, scale=scale, ratio=ratio)
            tl_in_E = _in_center_E(j1,      i1,         W, H, a)    # Top-Left
            br_in_E = _in_center_E(j1 + w1, i1 + h1,    W, H, a)    # Bottom-Right
            if (not tl_in_E) or (not br_in_E):
                break
        box1 = (j1, i1, j1 + w1, i1 + h1)

        # Generate Box 2
        ok = False
        for _ in range(box2_max_tries):
            i2, j2, h2, w2 = transforms.RandomResizedCrop.get_params(pil_img, scale=scale, ratio=ratio)
            box2 = (j2, i2, j2 + w2, i2 + h2)
            if _box_iou_xyxy(box1, box2) <= max_iou:
                ok = True
                break

        # If fallback
        if not ok:
            j2 = 0 if (j1 + w1 / 2) > (W / 2) else max(0, W - w1)
            i2 = 0 if (i1 + h1 / 2) > (H / 2) else max(0, H - h1)

            i2 = int(min(max(i2, 0), H - h1))
            j2 = int(min(max(j2, 0), W - w1))
            h2, w2 = h1, w1
            box2 = (j2, i2, j2 + w2, i2 + h2)

        # Crop & Resize
        A = TF.resized_crop(pil_img, top=i1, left=j1, height=h1, width=w1, size=size)
        B = TF.resized_crop(pil_img, top=i2, left=j2, height=h2, width=w2, size=size)
        A = self.random_hflip(A)
        B = self.random_hflip(B)

        # Random permute A, B
        if random.random() < 0.5:
            return A, B
        else:
            return B, A

    def generate_SSL_pair(self, view_A, view_B):
        # Generate training pair for Self-Supervised Learning via style augmentation
        params = get_style_augment_params()
        if random.random() < 0.5:   # Augmentation on view input
            view_inp = apply_style_augment_with_params(view_A, params)      # A*
            view_ref = view_B                                               # B
            view_tgt = view_A                                               # A
        else:                       # Augmentation on view refer & target
            view_inp = view_A                                               # A
            view_ref = apply_style_augment_with_params(view_B, params)      # B*
            view_tgt = apply_style_augment_with_params(view_A, params)      # A*

        return view_inp, view_ref, view_tgt


class Unseen_style_test_dataset(Dataset):
    def __init__(self, cfg, max_tries=10):

        self.max_tries = max_tries
        self.paths = []

        div2k_train_dir = os.path.join(cfg.dataset_root, 'Super_Resolution/DIV2K/DIV2K_train_HR')
        div2k_train_list = sorted(os.listdir(div2k_train_dir))
        self.paths.extend([os.path.join(div2k_train_dir, img_name) for img_name in div2k_train_list])

        div2k_valid_dir = os.path.join(cfg.dataset_root, 'Super_Resolution/DIV2K/DIV2K_valid_HR')
        div2k_valid_list = sorted(os.listdir(div2k_valid_dir))
        self.paths.extend([os.path.join(div2k_valid_dir, img_name) for img_name in div2k_valid_list])

        lsdir_root = os.path.join(cfg.dataset_root, 'Restoration/LSDIR')
        for lsdir_dir in os.listdir(lsdir_root):
            lsdir_list = sorted(os.listdir(os.path.join(lsdir_root, lsdir_dir)))
            lsdir_list = lsdir_list[int(len(lsdir_list) * 0.9):]
            self.paths.extend([os.path.join(lsdir_root, lsdir_dir, img_name) for img_name in lsdir_list])

        ppr10k_dir = os.path.join(cfg.dataset_root, 'Retouching/PPR10K/val/target_a_jpg')
        ppr10k_list = sorted(os.listdir(ppr10k_dir))
        self.paths.extend([os.path.join(ppr10k_dir, img_name) for img_name in ppr10k_list])

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def _to_tensor(self, pil_img):
        return self.to_tensor(pil_img).clamp(0., 1.)

    @torch.no_grad()
    def __getitem__(self, idx):

        path = self.paths[idx]
        src = Image.open(path).convert("RGB")

        view_A, view_B = self.get_two_crops_with_IoU_constraint(src, max_iou=0.)
        view_inp, view_ref, view_tgt = self.generate_SSL_pair(view_A, view_B)

        inp = self._to_tensor(view_inp)
        ref = self._to_tensor(view_ref)
        tgt = self._to_tensor(view_tgt)

        return {'canonicalizer_input': inp,                  # X
                'canonicalizer_GT': torch.zeros_like(tgt),   # Y_gt (= Y_0)
                'restyler_refer': ref,                       # R
                'restyler_GT': tgt,                          # Z_gt
                'canonical_refer': torch.zeros_like(ref),    # R_0
                'pair_type': 'uu'}

    def get_two_crops_with_IoU_constraint(self, pil_img,
                                          size=(448, 448), scale=(0.10, 0.36), ratio=(3/4, 4/3),
                                          max_iou=0.5, box1_max_tries=10, box2_max_tries=10, a=0.2):

        W, H = pil_img.size

        # Generate Box 1
        for _ in range(box1_max_tries):
            i1, j1, h1, w1 = transforms.RandomResizedCrop.get_params(pil_img, scale=scale, ratio=ratio)
            tl_in_E = _in_center_E(j1,      i1,         W, H, a)    # Top-Left
            br_in_E = _in_center_E(j1 + w1, i1 + h1,    W, H, a)    # Bottom-Right
            if (not tl_in_E) or (not br_in_E):
                break
        box1 = (j1, i1, j1 + w1, i1 + h1)

        # Generate Box 2
        ok = False
        for _ in range(box2_max_tries):
            i2, j2, h2, w2 = transforms.RandomResizedCrop.get_params(pil_img, scale=scale, ratio=ratio)
            box2 = (j2, i2, j2 + w2, i2 + h2)
            if _box_iou_xyxy(box1, box2) <= max_iou:
                ok = True
                break

        # If fallback
        if not ok:
            j2 = 0 if (j1 + w1 / 2) > (W / 2) else max(0, W - w1)
            i2 = 0 if (i1 + h1 / 2) > (H / 2) else max(0, H - h1)

            i2 = int(min(max(i2, 0), H - h1))
            j2 = int(min(max(j2, 0), W - w1))
            h2, w2 = h1, w1
            box2 = (j2, i2, j2 + w2, i2 + h2)

        # Crop & Resize
        A = TF.resized_crop(pil_img, top=i1, left=j1, height=h1, width=w1, size=size)
        B = TF.resized_crop(pil_img, top=i2, left=j2, height=h2, width=w2, size=size)

        # Random permute A, B
        if random.random() < 0.5:
            return A, B
        else:
            return B, A

    def generate_SSL_pair(self, view_A, view_B):
        # Generate training pair for Self-Supervised Learning via style augmentation
        params = get_style_augment_params()
        if random.random() < 0.5:   # Augmentation on view input
            view_inp = apply_style_augment_with_params(view_A, params)      # A*
            view_ref = view_B                                               # B
            view_tgt = view_A                                               # A
        else:                       # Augmentation on view refer & target
            view_inp = view_A                                               # A
            view_ref = apply_style_augment_with_params(view_B, params)      # B*
            view_tgt = apply_style_augment_with_params(view_A, params)      # A*

        return view_inp, view_ref, view_tgt


# ===================================================================================================
# For Known Style dataset (Style Library)
class FiveK_Known_style_train_dataset(Dataset):
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

        self.transform_plain = transforms.Compose([
            transforms.Resize(size=(448, 448)),
            transforms.ToTensor()])

        self.transform_input = transforms.Compose([
            transforms.Resize(size=(448, 448)),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)
            ], p=0.5),
            transforms.ToTensor()])

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
        A = self.n_anchors_per_style
        R = self.n_refers_per_anchor

        # anchor image sampling
        anchors_per_style = {}
        for style in self.style_ids:
            anchors_per_style[style] = random.sample(range(N), A)

        # Collect n_style * n_style * A * R data
        self.items = []
        for src_style in self.style_ids:
            anchors = anchors_per_style[src_style]
            for tgt_style in self.style_ids:
                for anchor_idx in anchors:
                    # get reference candidates per anchor image
                    candidates = self.ref_cand_indices[anchor_idx]

                    # reference image sampling
                    refers = random.sample(candidates, R)
                    for refer_idx in refers:
                        self.items.append((src_style, tgt_style, int(anchor_idx), int(refer_idx)))

    def __len__(self):
        return len(self.items)  # n_style * n_style * self.n_anchors_per_style * self.n_refers_per_anchor

    def __getitem__(self, idx):

        seed = random.randint(0, 2 ** 32)
        random.seed(seed)
        np.random.seed(seed)

        src_style, tgt_style, anchor_idx, refer_idx = self.items[idx]

        # Load paths
        canonicalizer_input_path = self.data_dict[src_style][anchor_idx]    # src style image of input content
        canonicalizer_gt_path    = self.data_dict['01'][anchor_idx]         # canonical image of input content
        restyler_refer_path      = self.data_dict[tgt_style][refer_idx]     # tgt style image of refer content
        restyler_gt_path         = self.data_dict[tgt_style][anchor_idx]    # tgt style image of input content
        canonical_refer_path     = self.data_dict['01'][refer_idx]          # canonical image of refer content

        # Load images
        canonicalizer_input = Image.open(canonicalizer_input_path).convert('RGB')
        canonicalizer_gt    = Image.open(canonicalizer_gt_path).convert('RGB')
        restyler_refer      = Image.open(restyler_refer_path).convert('RGB')
        restyler_gt         = Image.open(restyler_gt_path).convert('RGB')
        canonical_refer     = Image.open(canonical_refer_path).convert('RGB')

        # Apply same augmentations on same content
        crop1, flip1, params1 = self.get_aug_params(canonicalizer_input)
        canonicalizer_input = self.augment(canonicalizer_input, crop1, flip1, params1)
        canonicalizer_gt    = self.augment(canonicalizer_gt, crop1, flip1, params1)
        restyler_gt         = self.augment(restyler_gt, crop1, flip1, params1)

        crop2, flip2, params2 = self.get_aug_params(restyler_refer)
        restyler_refer = self.augment(restyler_refer, crop2, flip2, params2)
        canonical_refer = self.augment(canonical_refer, crop2, flip2, params2)

        # To tensor
        canonicalizer_input = self.transform_input(canonicalizer_input)
        canonicalizer_gt    = self.transform_plain(canonicalizer_gt)
        restyler_refer      = self.transform_plain(restyler_refer)
        restyler_gt         = self.transform_plain(restyler_gt)
        canonical_refer     = self.transform_plain(canonical_refer)

        return {'canonicalizer_input': canonicalizer_input, # X
                'canonicalizer_GT': canonicalizer_gt,       # Y_gt (= Y_0)
                'restyler_refer': restyler_refer,           # R
                'restyler_GT': restyler_gt,                 # Z_gt
                'canonical_refer': canonical_refer,         # R_0
                'pair_type': 'kk'}

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


class FiveK_Known_style_test_dataset(Dataset):
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

        name_list = sorted(name_list)
        random.seed(42)
        n_contents = cfg.testing['n_contents']
        sampled_ids = random.sample(name_list, n_contents)

        test_jsonl = os.path.join(root_dir, 'json/test.jsonl')
        top1_map = load_top1_map(test_jsonl)
        # topk_map = load_topk_map(test_jsonl)

        self.pairs = list()
        for src_style_idx, src_style_name in self.style_dicts.items():
            src_img_dir = os.path.join(root_dir, f'{src_style_idx}_{src_style_name}')

            for tgt_style_idx, tgt_style_name in self.style_dicts.items():

                if src_style_idx == tgt_style_idx:
                    continue        # Self style transition is not included in test

                if (src_style_idx == '01') or (tgt_style_idx == '01'):
                    continue        # Canonical style is not included in test

                tgt_img_dir = os.path.join(root_dir, f'{tgt_style_idx}_{tgt_style_name}')

                for input_image_name in sampled_ids:
                    # refer_list = topk_map.get(input_image_name)
                    # for refer_image_name in refer_list:
                    refer_image_name = top1_map.get(input_image_name)

                    input_img_path = os.path.join(src_img_dir, input_image_name)
                    refer_img_path = os.path.join(tgt_img_dir, refer_image_name)
                    canonical_gt_path = os.path.join(canonical_dir, input_image_name)
                    restyler_gt_path = os.path.join(tgt_img_dir, input_image_name)
                    self.pairs.append((int(src_style_idx), int(tgt_style_idx),
                                       input_img_path, refer_img_path,
                                       canonical_gt_path, restyler_gt_path))

        self.transform = transforms.Compose([transforms.Resize(size=(448, 448)),
                                             transforms.ToTensor()])

    def __getitem__(self, idx):
        src_style, tgt_style, input_img_path, refer_img_path, canonical_gt_path, restyler_gt_path = self.pairs[idx]
        tag = (src_style, tgt_style)
        return {'tag': tag, 'img_name': os.path.basename(input_img_path),
                'canonicalizer_input': self.load_image(input_img_path),
                'canonicalizer_GT': self.load_image(canonical_gt_path),
                'restyler_refer': self.load_image(refer_img_path),
                'restyler_GT': self.load_image(restyler_gt_path)}

    def __len__(self):
        return len(self.pairs)

    def load_image(self, path):
        img = Image.open(path).convert("RGB")
        return self.transform(img)


class FiveK_Known_style_eval_dataset(Dataset):
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

        name_list = sorted(name_list)
        random.seed(42)
        n_contents = cfg.testing['n_contents']
        sampled_ids = random.sample(name_list, n_contents)

        test_jsonl = os.path.join(root_dir, 'json/test.jsonl')
        top1_map =  load_top1_map(test_jsonl)
        # topk_map = load_topk_map(test_jsonl)

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
                    # refer_list = topk_map.get(input_image_name)
                    # for refer_image_name in refer_list:
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
        src_style, tgt_style, input_img_path, refer_img_path, canonical_gt_path, restyler_gt_path, canonical_refer_path = self.pairs[idx]
        tag = (src_style, tgt_style)

        return {'tag': tag, 'img_name': os.path.basename(input_img_path),
                'input_img_path': input_img_path,
                'refer_img_path': refer_img_path,
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



# ===================================================================================================
# For Evaluation datasets
class Supervised_dataset_eval(Dataset):
    def __init__(self, dataset_root):

        self.style_dicts = style_dicts

        root_dir = os.path.join(dataset_root, 'Style_transition')

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

        name_list = sorted(name_list)
        random.seed(42)
        n_contents = 100
        sampled_ids = random.sample(name_list, n_contents)

        test_jsonl = os.path.join(root_dir, 'json/test.jsonl')
        top1_map =  load_top1_map(test_jsonl)

        self.pairs = list()
        for src_style_idx, src_style_name in self.style_dicts.items():
            src_img_dir = os.path.join(root_dir, f'{src_style_idx}_{src_style_name}')

            for tgt_style_idx, tgt_style_name in self.style_dicts.items():

                if src_style_idx == tgt_style_idx:
                    continue        # Self style transition is not included in test

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
        src_style, tgt_style, input_img_path, refer_img_path, canonical_gt_path, restyler_gt_path, canonical_refer_path = self.pairs[idx]
        tag = (src_style, tgt_style)

        # # Augmentation on view input
        # params = get_style_augment_params()
        # inp = Image.open(input_img_path).convert("RGB")
        # ref = Image.open(refer_img_path).convert("RGB")
        # inp_aug = apply_style_augment_with_params(inp, params)  # A*
        # ref_aug = apply_style_augment_with_params(ref, params)  # A*

        return {'tag': tag, 'img_name': os.path.basename(input_img_path),
                'inp': self.load_image(input_img_path),
                'ref': self.load_image(refer_img_path),
                # 'inp_aug': self.transform(inp_aug),
                # 'ref_aug': self.transform(ref_aug),
                'gt': self.load_image(restyler_gt_path),
                'dataset_name': 'Supervised'}

    def __len__(self):
        return len(self.pairs)

    def load_image(self, path):
        img = Image.open(path).convert("RGB")
        return self.transform(img)


class Unsupervised_dataset_eval(Dataset):
    def __init__(self, dataset_root):

        self.paths = []
        self.dataset_names = []

        # load LSDIR
        lsdir_root = os.path.join(dataset_root, 'Restoration/LSDIR')
        for lsdir_dir in os.listdir(lsdir_root):
            lsdir_list = sorted(os.listdir(os.path.join(lsdir_root, lsdir_dir)))
            lsdir_list = lsdir_list[int(len(lsdir_list) * 0.9):]
            self.paths.extend([os.path.join(lsdir_root, lsdir_dir, img_name) for img_name in lsdir_list])
            self.dataset_names.extend(['LSDIR' for _ in range(len(lsdir_list))])

        # load DIV2K
        div2k_train_dir = os.path.join(dataset_root, 'Super_Resolution/DIV2K/DIV2K_train_HR')
        div2k_train_list = sorted(os.listdir(div2k_train_dir))
        div2k_valid_dir = os.path.join(dataset_root, 'Super_Resolution/DIV2K/DIV2K_valid_HR')
        div2k_valid_list = sorted(os.listdir(div2k_valid_dir))
        self.paths.extend([os.path.join(div2k_train_dir, img_name) for img_name in div2k_train_list])
        self.paths.extend([os.path.join(div2k_valid_dir, img_name) for img_name in div2k_valid_list])
        self.dataset_names.extend(['DIV2K' for _ in range(len(div2k_train_list))])
        self.dataset_names.extend(['DIV2K' for _ in range(len(div2k_valid_list))])

        # load PPR10K
        ppr10k_dir = os.path.join(dataset_root, 'Retouching/PPR10K/val/target_a_jpg')
        ppr10k_list = sorted(os.listdir(ppr10k_dir))
        self.paths.extend([os.path.join(ppr10k_dir, img_name) for img_name in ppr10k_list])
        self.dataset_names.extend(['PPR10K' for _ in range(len(ppr10k_list))])

        # load Food-101
        food101_dir = os.path.join(dataset_root, 'food-101/subset/images')
        food101_list = sorted(os.listdir(food101_dir))
        self.paths.extend([os.path.join(food101_dir, img_name) for img_name in food101_list])
        self.dataset_names.extend(['FOOD101' for _ in range(len(food101_list))])

        # load GLD-v2
        gldv2_dir = os.path.join(dataset_root, 'GLD-v2/subset/images')
        gldv2_list = sorted(os.listdir(gldv2_dir))
        self.paths.extend([os.path.join(gldv2_dir, img_name) for img_name in gldv2_list])
        self.dataset_names.extend(['GLDv2' for _ in range(len(gldv2_list))])

        self.max_tries = 10
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def _to_tensor(self, pil_img):
        return self.to_tensor(pil_img).clamp(0., 1.)

    @torch.no_grad()
    def __getitem__(self, idx):

        path = self.paths[idx]
        src = Image.open(path).convert("RGB")

        view_A, view_B = self.get_two_crops_with_IoU_constraint(src, max_iou=0.)
        view_inp, view_ref, view_tgt = self.generate_SSL_pair(view_A, view_B)

        inp = self._to_tensor(view_inp)
        ref = self._to_tensor(view_ref)
        tgt = self._to_tensor(view_tgt)

        return {'inp': inp,                 # X
                'ref': ref,                 # R
                'gt': tgt,                  # Z_gt
                'dataset_name': self.dataset_names[idx]}

    def get_two_crops_with_IoU_constraint(self, pil_img,
                                          size=(448, 448), scale=(0.10, 0.36), ratio=(3/4, 4/3),
                                          max_iou=0.5, box1_max_tries=10, box2_max_tries=10, a=0.2):

        W, H = pil_img.size

        # Generate Box 1
        for _ in range(box1_max_tries):
            i1, j1, h1, w1 = transforms.RandomResizedCrop.get_params(pil_img, scale=scale, ratio=ratio)
            tl_in_E = _in_center_E(j1,      i1,         W, H, a)    # Top-Left
            br_in_E = _in_center_E(j1 + w1, i1 + h1,    W, H, a)    # Bottom-Right
            if (not tl_in_E) or (not br_in_E):
                break
        box1 = (j1, i1, j1 + w1, i1 + h1)

        # Generate Box 2
        ok = False
        for _ in range(box2_max_tries):
            i2, j2, h2, w2 = transforms.RandomResizedCrop.get_params(pil_img, scale=scale, ratio=ratio)
            box2 = (j2, i2, j2 + w2, i2 + h2)
            if _box_iou_xyxy(box1, box2) <= max_iou:
                ok = True
                break

        # If fallback
        if not ok:
            j2 = 0 if (j1 + w1 / 2) > (W / 2) else max(0, W - w1)
            i2 = 0 if (i1 + h1 / 2) > (H / 2) else max(0, H - h1)

            i2 = int(min(max(i2, 0), H - h1))
            j2 = int(min(max(j2, 0), W - w1))
            h2, w2 = h1, w1
            box2 = (j2, i2, j2 + w2, i2 + h2)

        # Crop & Resize
        A = TF.resized_crop(pil_img, top=i1, left=j1, height=h1, width=w1, size=size)
        B = TF.resized_crop(pil_img, top=i2, left=j2, height=h2, width=w2, size=size)

        # Random permute A, B
        return A, B

    def generate_SSL_pair(self, view_A, view_B):
        # Generate training pair for Self-Supervised Learning via style augmentation
        params = get_style_augment_params()

        # Augmentation on view input
        view_inp = apply_style_augment_with_params(view_A, params)  # A*
        view_ref = view_B                                           # B
        view_tgt = view_A                                           # A

        return view_inp, view_ref, view_tgt


