import os
import math
import time
import torch
import numpy as np

from utils.util import to_np, save_best_model, logger
from utils.viz_utils import Visualizer
from utils.calculate_metrics import Evaluator, Evaluator_SupCon


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
        src_style, tgt_style = batch['tag'][0], batch['tag'][1]
        img_name = batch['img_name']
        model_input = batch['canonicalizer_input'].cuda()
        model_refer = batch['restyler_refer'].cuda()
        canonicalizer_GT = batch['canonicalizer_GT'].cuda()
        restyler_GT = batch['restyler_GT'].cuda()

        # Forward model
        with torch.no_grad():
            outputs = model(model_input, model_refer)

        n += len(model_input)
        psnr += eval_tools.measure_PSNR(outputs['restyled'], restyler_GT, reduction='sum')

    PSNR = psnr / n
    print('%s Test ==> mPSNR %5f' % (cfg.dataset_name, PSNR))

    now = {'PSNR': PSNR}
    best = save_best_model(cfg, model, epoch, now, best, metric='PSNR')
    return best, now


def analysis(cfg, model, test_loader):
    model.eval()

    viz_tools = Visualizer()
    eval_tools = Evaluator(cfg)
    style_dicts = test_loader.dataset.style_dicts
    style_ids = list(style_dicts.keys())

    num_dict = {(src_s, tgt_s): 0 for src_s in style_ids for tgt_s in style_ids}
    PSNR_dict = {(src_s, tgt_s): 0 for src_s in style_ids for tgt_s in style_ids}

    with torch.no_grad():
        torch.cuda.empty_cache()

        for i, batch in enumerate(test_loader):
            print('Processing [%04d/%04d]...' % (i, len(test_loader)), end='\r')

            # Load data
            src_style, tgt_style = batch['tag'][0], batch['tag'][1]
            img_name = batch['img_name']
            model_input = batch['canonicalizer_input'].cuda()
            model_refer = batch['restyler_refer'].cuda()
            canonicalizer_GT = batch['canonicalizer_GT'].cuda()
            restyler_GT = batch['restyler_GT'].cuda()
            canonical_refer = batch['canonical_refer'].cuda()

            # Forward model
            outputs = model(model_input, model_refer)

            # Evaluation
            psnr = eval_tools.measure_PSNR(outputs['restyled'], restyler_GT, reduction=None)

            for b_idx in range(len(model_input)):
                src_style_b, tgt_style_b = src_style[b_idx], tgt_style[b_idx]
                num_dict[(f'{src_style_b:02d}', f'{tgt_style_b:02d}')] += 1
                PSNR_dict[(f'{src_style_b:02d}', f'{tgt_style_b:02d}')] += psnr[b_idx]

            if cfg.viz:
                for b_idx in range(len(model_input)):
                    src_style_b, tgt_style_b = src_style[b_idx], tgt_style[b_idx]
                    if src_style_b == tgt_style_b:
                        continue
                    tag = f'{src_style_b}_to_{tgt_style_b}'
                    viz_contents = {'input': model_input[b_idx],
                                    'canonicalized': outputs['canonicalized'][b_idx],
                                    'canonicalizer_GT': canonicalizer_GT[b_idx],
                                    'restyler_refer': model_refer[b_idx],
                                    'restyler_GT': restyler_GT[b_idx],
                                    'restyled': outputs['restyled'][b_idx],
                                    'canonical_refer': canonical_refer[b_idx]}
                    viz_tools.viz_analysis(cfg, viz_contents, tag=tag, img_name=img_name[b_idx])

        print('')
        PSNR_matrix = {(src_s, tgt_s): PSNR_dict[(src_s, tgt_s)] / num_dict[(src_s, tgt_s)]
                       for src_s in style_ids for tgt_s in style_ids if num_dict[(src_s, tgt_s)] != 0}

        mPSNR = sum(PSNR_matrix.values()) / len(PSNR_matrix.values())

        print(f'{cfg.dataset_name} Restyle ==> mPSNR : {mPSNR:.5f} dB')
        logger(f'[{cfg.dataset_name}] ==> mPSNR : {mPSNR:.5f} dB \n', f'{cfg.save_dir}/stage2_results.txt')


def batch_reformatting(img, gt, style_idx):
    gt = gt.squeeze(0)
    img = img.repeat(gt.shape[0], 1, 1, 1)
    style_idx = style_idx.squeeze(0)
    return img, gt, style_idx
