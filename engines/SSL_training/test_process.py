import copy
import torch
from torch.utils.data import DataLoader
from dataloaders.Style_Library.SSL_training import Unseen_style_test_dataset, FiveK_Known_style_eval_dataset
from engines.SSL_training.forward_test import analysis_kk, analysis_uu


def test_process(cfg, model):
    checkpoint = torch.load(cfg.checkpoint['model'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    checkpoint = None

    model.cuda()

    kk_test_dataset = FiveK_Known_style_eval_dataset(cfg)
    uu_test_dataset = Unseen_style_test_dataset(cfg)

    g = torch.Generator()
    g.manual_seed(42)
    kk_test_loader = DataLoader(kk_test_dataset, batch_size=20, num_workers=cfg.dataset['num_workers'], shuffle=False, generator=g)
    uu_test_loader = DataLoader(uu_test_dataset, batch_size=20, num_workers=cfg.dataset['num_workers'], shuffle=False, generator=g)

    analysis_kk(cfg, model, kk_test_loader)
    analysis_uu(cfg, model, uu_test_loader)
