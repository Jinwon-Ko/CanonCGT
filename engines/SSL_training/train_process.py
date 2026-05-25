import copy
import torch
from torch.utils.data import DataLoader
from dataloaders.Style_Library.SSL_training import (Unseen_style_train_dataset, Unseen_style_test_dataset,
                                                    FiveK_Known_style_train_dataset, FiveK_Known_style_test_dataset,
                                                    make_mixed_train_loader)

from utils.util import save_best_model
from engines.utils_train import define_losses_and_metrics, update_dict, update_LRs, save_losses_and_performances
from engines.SSL_training.forward_train import train_one_epoch_mixed
from engines.SSL_training.forward_test import evaluation


def train_process(cfg, model, criterion, optimizer, lr_scheduler):
    checkpoint = torch.load(cfg.checkpoint['model'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    start_epoch = checkpoint['epoch'] if cfg.load else 0
    checkpoint = None

    # (unknown, unknown) paired dataset
    uu_train_dataset = Unseen_style_train_dataset(cfg)
    uu_test_dataset = Unseen_style_test_dataset(cfg)

    # (known, known) paired dataset
    kk_train_dataset = FiveK_Known_style_train_dataset(cfg)
    kk_test_dataset = FiveK_Known_style_test_dataset(cfg)

    # Define dataloader
    g = torch.Generator()
    g.manual_seed(42)
    mixed_train_loader = make_mixed_train_loader(cfg, kk_train_dataset, uu_train_dataset)
    uu_test_loader = DataLoader(uu_test_dataset, batch_size=20, num_workers=cfg.dataset['num_workers'], shuffle=False, generator=g)
    kk_test_loader = DataLoader(kk_test_dataset, batch_size=20, num_workers=cfg.dataset['num_workers'], shuffle=False, generator=g)

    model.cuda()
    criterion.cuda()

    Losses, Performances, best, now = define_losses_and_metrics(cfg)
    LRs = []

    len_uu = len(uu_test_dataset.paths)
    len_kk = len(kk_test_dataset.pairs)
    if cfg.load:
        PSNR_uu = evaluation(cfg, start_epoch - 1, model, uu_test_loader, pair_type='uu')
        PSNR_kk = evaluation(cfg, start_epoch - 1, model, kk_test_loader, pair_type='kk')
        PSNR = (PSNR_uu * len_uu + PSNR_kk * len_kk) / (len_uu + len_kk)
        now = {'PSNR': PSNR}
        best = save_best_model(cfg, model, start_epoch - 1, now, best, metric='PSNR')
        lr_scheduler.step(start_epoch - 1)

    for epoch in range(start_epoch, cfg.training['epochs']):
        lr_scheduler.step(epoch)
        kk_train_dataset.build_epoch()
        LRs = update_LRs(LRs, optimizer.param_groups[0]['lr'])
        model, optimizer, losses = train_one_epoch_mixed(cfg, epoch, model, mixed_train_loader, criterion, optimizer)
        PSNR_uu = evaluation(cfg, epoch, model, uu_test_loader, pair_type='uu')
        PSNR_kk = evaluation(cfg, epoch, model, kk_test_loader, pair_type='kk')

        PSNR = (PSNR_uu * len_uu + PSNR_kk * len_kk) / (len_uu + len_kk)
        now = {'PSNR': PSNR}
        best = save_best_model(cfg, model, epoch, now, best, metric='PSNR')

        Losses, Performances = update_dict(Losses, losses, Performances, now)
        save_losses_and_performances(cfg, Losses, Performances['PSNR'], LRs)

    print('best_Performances : ', best)


