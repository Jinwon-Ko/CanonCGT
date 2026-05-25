import os
import math
import time
import torch
import numpy as np

from utils.util import to_np, save_best_model
from utils.viz_utils import Visualizer
from utils.calculate_metrics import Evaluator



def evaluation(cfg, epoch, model, test_loader, best, now):
    if (epoch + 1) < cfg.testing['start_eval_epoch']:
        return best, now
    
    if (epoch + 1) % cfg.testing['eval_epoch'] != 0:
        return best, now

    model.eval()

    viz_tools = Visualizer()
    eval_tools = Evaluator(cfg)

    n = 0
    psnr = 0
    for i, batch in enumerate(test_loader):
        print('[Epoch: %d][%d/%d]' % (epoch, i, len(test_loader)), end='\r')

        # Load data
        imgs = batch['img'].cuda()
        gt = batch['gt'].cuda()
        style_indices = batch['style_idx']
        img_name = batch['img_name']

        imgs, gts, style_indices = batch_reformatting(imgs, gt, style_indices)

        with torch.no_grad():
            outputs = model(imgs, style_indices)

        n += len(imgs)
        psnr += eval_tools.measure_PSNR(outputs['canonicalized'], gts)

    PSNR = psnr / n
    print('%s Test ==> PSNR %5f' % (cfg.dataset_name, PSNR))

    now = {'PSNR': PSNR}
    best = save_best_model(cfg, model, epoch, now, best, metric='PSNR')
    return best, now


def analysis(cfg, model, test_loader):
    model.eval()

    viz_tools = Visualizer()
    eval_tools = Evaluator(cfg)

    style_dicts = test_loader.dataset.style_dicts
    psnr_per_expert = {style_idx: 0 for style_idx in style_dicts}
    with torch.no_grad():
        torch.cuda.empty_cache()

        for i, batch in enumerate(test_loader):
            print('Processing [%04d/%04d]...' % (i, len(test_loader)), end='\r')

            # Load data
            imgs = batch['img'].cuda()
            gt = batch['gt'].cuda()
            style_indices = batch['style_idx']
            img_name = batch['img_name']

            imgs, gts, style_indices = batch_reformatting(imgs, gt, style_indices)

            outputs = model(imgs, style_indices)

            # Evaluation
            PSNRs = eval_tools.measure_PSNR(outputs['canonicalized'], gts, reduction=None)
            for k in range(len(imgs)):
                style_idx = int(style_indices[k])
                psnr_per_expert[f'{style_idx:02d}'] += PSNRs[k]

            # Visualize
            if cfg.viz:
                viz_tools.update_image(gts[0], name='gt')
                viz_tools.update_image(torch.zeros_like(gts[0]), name='empty')
                for k in range(len(imgs)):
                    # Visualize
                    style_idx = int(style_indices[k])
                    viz_tools.update_image(imgs[k], name=f'input_{style_idx:02d}')  # Model input
                    viz_tools.update_image(outputs['canonicalized'][k], name=f'output_{style_idx:02d}')  # Model output

                viz_tools.viz_comparison_per_style_imgs(cfg, img_name=img_name[0])  # Model inputs
                viz_tools.viz_comparison_per_style_preds(cfg, img_name=img_name[0])  # Model outputs

        max_len = max([len(f'style_{idx}_{style_dicts[idx]}') for idx in psnr_per_expert])

        total_PSNR = 0
        for style_idx, psnr in psnr_per_expert.items():
            PSNR = psnr / len(test_loader)
            style_name = f'style_{style_idx}_{style_dicts[style_idx]}'
            print(f'{cfg.dataset_name} / {style_name:<{max_len}} ==> [PSNR {PSNR:.5f}]')

            total_PSNR += PSNR

        total_PSNR = total_PSNR / len(psnr_per_expert.keys())
        print(f'{cfg.dataset_name} ==> [Total PSNR {total_PSNR:.5f}]')


def batch_reformatting(img, gt, style_idx):
    img = img.squeeze(0)
    gt = gt.repeat(img.shape[0], 1, 1, 1)
    style_indices = [idx[0] for idx in style_idx]
    return img, gt, style_indices
