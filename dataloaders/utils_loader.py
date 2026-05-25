import json
import torch
import random
import torchvision.transforms.functional as TF


def load_top1_map(jsonl_path):
    top1_map = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            img_id = rec.get('img_id')
            top1 = rec.get('top1')
            ref_id = top1.get('ref_id')
            top1_map[img_id] = ref_id
    return top1_map


def load_topk_map(jsonl_path):
    topk_map = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            img_id = rec.get('img_id')
            topk = rec.get('topk')
            ref_ids = [item['ref_id'] for item in topk]
            topk_map[img_id] = ref_ids
    return topk_map


def gen_random_aug_params(value, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
    if value < 0:
        raise ValueError('Must be non negative.')

    # Define interval
    value_range = [center - float(value), center + float(value)]

    if clip_first_on_zero:
        value_range[0] = max(value_range[0], 0.0)

    # Bound check
    if not bound[0] <= value_range[0] <= value_range[1] <= bound[1]:
        raise ValueError(f"Values should be between {bound}")

    # Uniform sampling
    param = float(torch.empty(1).uniform_(value_range[0], value_range[1]))
    return param


def gen_random_aug_params_min_max(min_value, max_value, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
    if min_value < 0 or max_value < 0:
        raise ValueError("min_value and max_value must be non-negative.")
    if min_value > max_value:
        raise ValueError("min_value must not exceed max_value.")

    # Define interval
    left_range  = [center - float(max_value), center - float(min_value)]
    right_range = [center + float(min_value), center + float(max_value)]

    if clip_first_on_zero:
        left_range[0] = max(left_range[0], 0.0)

    # Bound check
    for rng in (left_range, right_range):
        if not bound[0] <= rng[0] <= rng[1] <= bound[1]:
            raise ValueError(f"Values should be between {bound}, but got {rng}")

    # Select interval
    chosen_range = random.choice([left_range, right_range])

    # Uniform sampling
    param = float(torch.empty(1).uniform_(chosen_range[0], chosen_range[1]))
    return param


def apply_style_augment_to_tensor(pil_img):
    # brightness  = gen_random_aug_params(value=0.4, center=1, bound=(0, float("inf")), clip_first_on_zero=True)
    # contrast    = gen_random_aug_params(value=0.4, center=1, bound=(0, float("inf")), clip_first_on_zero=True)
    # saturation  = gen_random_aug_params(value=0.4, center=1, bound=(0, float("inf")), clip_first_on_zero=True)
    # hue         = gen_random_aug_params(value=0.1, center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
    brightness  = gen_random_aug_params_min_max(min_value=0.1, max_value=0.4, center=1, bound=(0, float("inf")), clip_first_on_zero=True)
    contrast    = gen_random_aug_params_min_max(min_value=0.1, max_value=0.4, center=1, bound=(0, float("inf")), clip_first_on_zero=True)
    saturation  = gen_random_aug_params_min_max(min_value=0.1, max_value=0.4, center=1, bound=(0, float("inf")), clip_first_on_zero=True)
    hue         = gen_random_aug_params_min_max(min_value=0.01, max_value=0.05, center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    pil_img = TF.adjust_brightness(pil_img, brightness)
    pil_img = TF.adjust_contrast(pil_img, contrast)
    pil_img = TF.adjust_saturation(pil_img, saturation)
    pil_img = TF.adjust_hue(pil_img, hue)
    return pil_img


def get_style_augment_params():
    # brightness  = gen_random_aug_params(value=0.4, center=1, bound=(0, float("inf")), clip_first_on_zero=True)
    # contrast    = gen_random_aug_params(value=0.4, center=1, bound=(0, float("inf")), clip_first_on_zero=True)
    # saturation  = gen_random_aug_params(value=0.4, center=1, bound=(0, float("inf")), clip_first_on_zero=True)
    # hue         = gen_random_aug_params(value=0.1, center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
    brightness  = gen_random_aug_params_min_max(min_value=0.1, max_value=0.4, center=1, bound=(0, float("inf")), clip_first_on_zero=True)
    contrast    = gen_random_aug_params_min_max(min_value=0.1, max_value=0.4, center=1, bound=(0, float("inf")), clip_first_on_zero=True)
    saturation  = gen_random_aug_params_min_max(min_value=0.1, max_value=0.4, center=1, bound=(0, float("inf")), clip_first_on_zero=True)
    hue         = gen_random_aug_params_min_max(min_value=0.01, max_value=0.05, center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
    return brightness, contrast, saturation, hue


def apply_style_augment_with_params(pil_img, params):
    brightness, contrast, saturation, hue = params
    pil_img = TF.adjust_brightness(pil_img, brightness)
    pil_img = TF.adjust_contrast(pil_img, contrast)
    pil_img = TF.adjust_saturation(pil_img, saturation)
    pil_img = TF.adjust_hue(pil_img, hue)
    return pil_img



def _in_center_E(x, y, W, H, a=0.2):
    return (a*W <= x < (1.0-a)*W) and (a*H <= y < (1.0-a)*H)


def _box_iou_xyxy(box1, box2):
    # b = (x1, y1, x2, y2) in absolute pixels
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter + 1e-12
    return inter / union
