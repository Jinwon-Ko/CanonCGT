import os
import math
import time
import torch
import numpy as np

from utils.util import to_np, logger
from utils.viz_utils import Visualizer
from utils.calculate_metrics import Evaluator, Evaluator_SupCon


def evaluation(cfg, epoch, model, test_loader, pair_type='kk'):
    if (epoch + 1) < cfg.testing['start_eval_epoch']:
        return 0

    if (epoch + 1) % cfg.testing['eval_epoch'] != 0:
        return 0

    model.eval()

    viz_tools = Visualizer()
    eval_tools = Evaluator(cfg)

    n = 0
    psnr = 0
    for i, batch in enumerate(test_loader):
        print('[Epoch: %d][%d/%d]' % (epoch, i, len(test_loader)), end='\r')

        # Load data
        model_input = batch['canonicalizer_input'].cuda()
        model_refer = batch['restyler_refer'].cuda()
        restyler_GT = batch['restyler_GT'].cuda()

        # Forward model
        with torch.no_grad():
            outputs = model(model_input, model_refer)

        n += len(model_input)
        psnr += eval_tools.measure_PSNR(outputs['restyled'], restyler_GT, reduction='sum')

    PSNR = psnr / n
    print(f'Eval on {pair_type.upper()} paired dataset ==> mPSNR {PSNR:.5f}')
    return PSNR


def evaluation_kk(cfg, model, eval_loader):
    model.eval()

    viz_tools = Visualizer()
    eval_tools = Evaluator(cfg)
    eval_tools.cuda().eval()

    style_dicts = eval_loader.dataset.style_dicts
    style_ids = list(style_dicts.keys())

    num = 0
    PSNR, SSIM = 0, 0
    LPIPS, SSIM_ED = 0, 0
    GRAM, H_Corr, H_Chi = 0, 0, 0

    with torch.no_grad():
        torch.cuda.empty_cache()

        for i, batch in enumerate(eval_loader):
            print('Processing [%04d/%04d]...' % (i, len(eval_loader)), end='\r')

            # Load data
            img_name = batch['img_name']
            src_style, tgt_style = batch['tag'][0], batch['tag'][1]
            inp = batch['canonicalizer_input'].cuda()
            ref = batch['restyler_refer'].cuda()
            restyler_GT = batch['restyler_GT'].cuda()

            # Forward model
            outputs = model(inp, ref)

            # Evaluation
            # Fidelity (GT <-> Output)
            PSNR += eval_tools.measure_PSNR(outputs['restyled'], restyler_GT, reduction='sum')
            SSIM += eval_tools.measure_SSIM(outputs['restyled'], restyler_GT, reduction='sum')

            # Content (input <-> Output)
            LPIPS += eval_tools.measure_LPIPS(outputs['restyled'], inp, reduction='sum')
            SSIM_ED += eval_tools.measure_SSIM_ED(outputs['restyled'], inp, reduction='sum')

            # Style (refer <-> Output)
            GRAM += eval_tools.measure_Gram(outputs['restyled'], ref, reduction='sum')
            H_Corr += eval_tools.measure_HCorr(outputs['restyled'], ref, reduction='sum')
            H_Chi += eval_tools.measure_HChi(outputs['restyled'], ref, reduction='sum')

            num += len(inp)

        mPSNR = PSNR / num; mSSIM = SSIM / num
        mLPIPS = LPIPS / num; mSSIM_ED = SSIM_ED / num
        mGRAM = GRAM / num; mH_Corr = H_Corr / num; mH_Chi = H_Chi / num

        print(f'\nEval on paired dataset ==> Fidelity \t: [PSNR {mPSNR:.5f}  SSIM {mSSIM:.5f}]')
        print(f'Eval on paired dataset ==> Content \t: [LPIPS {mLPIPS:.5f}  SSIM_ED {mSSIM_ED:.5f}]')
        print(f'Eval on paired dataset ==> Style \t: [GRAM {mGRAM:.5f}  H_Corr {mH_Corr:.5f}  H_Chi {mH_Chi:.5f}]')


def evaluation_uu(cfg, model, eval_loader):
    model.eval()

    viz_tools = Visualizer()
    eval_tools = Evaluator(cfg)
    eval_tools.cuda().eval()

    num = 0
    LPIPS, SSIM_ED = 0, 0
    GRAM, H_Corr, H_Chi = 0, 0, 0
    for i, batch in enumerate(eval_loader):
        print('Processing [%04d/%04d]...' % (i, len(eval_loader)), end='\r')

        # Load data
        inp = batch['input'].cuda()
        ref = batch['refer'].cuda()

        # Forward model
        with torch.no_grad():
            outputs = model(inp, ref)

        # Evaluation
        # Content (input <-> Output)
        LPIPS += eval_tools.measure_LPIPS(outputs['restyled'], inp, reduction='sum')
        SSIM_ED += eval_tools.measure_SSIM_ED(outputs['restyled'], inp, reduction='sum')

        # Style (refer <-> Output)
        GRAM += eval_tools.measure_Gram(outputs['restyled'], ref, reduction='sum')
        H_Corr += eval_tools.measure_HCorr(outputs['restyled'], ref, reduction='sum')
        H_Chi += eval_tools.measure_HChi(outputs['restyled'], ref, reduction='sum')

        num += len(inp)

    mLPIPS = LPIPS / num; mSSIM_ED = SSIM_ED / num
    mGRAM = GRAM / num; mH_Corr = H_Corr / num; mH_Chi = H_Chi / num

    print(f'\nEval on unpaired dataset ==> Content \t: [LPIPS {mLPIPS:.5f}  SSIM_ED {mSSIM_ED:.5f}]')
    print(f'Eval on unpaired dataset ==> Style \t: [GRAM {mGRAM:.5f}  H_Corr {mH_Corr:.5f}  H_Chi {mH_Chi:.5f}]')

