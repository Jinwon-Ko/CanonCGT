import copy
import torch
from torch.utils.data import DataLoader
from dataloaders.Style_Library.SSL_training import FiveK_Known_style_eval_dataset, Unseen_style_test_dataset
from engines.SSL_training.forward_eval import evaluation_kk, evaluation_uu


def eval_process(cfg, model):
    checkpoint = torch.load(cfg.checkpoint['model'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    checkpoint = None

    model.cuda()

    kk_eval_dataset = FiveK_Known_style_eval_dataset(cfg)
    uu_eval_dataset = Unseen_style_test_dataset(cfg)

    g = torch.Generator()
    g.manual_seed(42)
    kk_eval_loader = DataLoader(kk_eval_dataset, batch_size=1, num_workers=cfg.dataset['num_workers'], shuffle=False, generator=g)
    uu_eval_loader = DataLoader(uu_eval_dataset, batch_size=1, num_workers=cfg.dataset['num_workers'], shuffle=False, generator=g)

    evaluation_kk(cfg, model, kk_eval_loader)
    evaluation_uu(cfg, model, uu_eval_loader)
