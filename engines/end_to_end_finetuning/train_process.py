import torch
from torch.utils.data import DataLoader

from dataloaders.factory import load_dataset
from engines.utils_train import define_losses_and_metrics, update_dict, update_LRs, save_losses_and_performances
from engines.end_to_end_finetuning.forward_train import train_one_epoch
from engines.end_to_end_finetuning.forward_test import evaluation


def train_process(cfg, model, criterion, optimizer, lr_scheduler):
    train_dataset = load_dataset(cfg, mode='train')
    test_dataset = load_dataset(cfg, mode='test')
    batch_size = cfg.dataset['n_style_per_batch'] * cfg.dataset['n_content_per_batch']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=cfg.dataset['num_workers'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=20, num_workers=cfg.dataset['num_workers'], shuffle=False)

    start_epoch = 0
    Losses, Performances, best, now = define_losses_and_metrics(cfg)
    LRs = []

    if cfg.load:
        checkpoint = torch.load(cfg.checkpoint['model'])
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        checkpoint = None
    else:
        from collections import OrderedDict
        ckpt_dict = OrderedDict()

        checkpoint = torch.load(cfg.checkpoint['Embedding_network'])
        ckpt_dict.update(checkpoint['model_state_dict'])
        checkpoint = torch.load(cfg.checkpoint['Canonicalizer'])
        ckpt_dict.update(checkpoint['model_state_dict'])
        checkpoint = torch.load(cfg.checkpoint['Restyler'])
        ckpt_dict.update(checkpoint['model_state_dict'])
        checkpoint = None

        model.load_state_dict(ckpt_dict)

    model.cuda()
    criterion.cuda()

    if cfg.load:
        best, now = evaluation(cfg, start_epoch - 1, model, test_loader, best, now)
        lr_scheduler.step(start_epoch - 1)

    for epoch in range(start_epoch, cfg.training['epochs']):
        lr_scheduler.step(epoch)
        train_dataset.build_epoch()
        LRs = update_LRs(LRs, optimizer.param_groups[0]['lr'])
        model, optimizer, losses = train_one_epoch(cfg, epoch, model, train_loader, criterion, optimizer)
        best, now = evaluation(cfg, epoch, model, test_loader, best, now)

        Losses, Performances = update_dict(Losses, losses, Performances, now)
        save_losses_and_performances(cfg, Losses, Performances['PSNR'], LRs)

    print('best_Performances : ', best)


