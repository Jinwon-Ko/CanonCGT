import torch

from torch.utils.data import DataLoader
from dataloaders.factory import load_dataset
from engines.canonicalizer.forward_test import analysis


def test_process(cfg, model):
    if cfg.load:
        checkpoint = torch.load(cfg.checkpoint['model'])
        model.load_state_dict(checkpoint['model_state_dict'])
        checkpoint = None

    model.cuda()

    test_dataset = load_dataset(cfg, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=cfg.dataset['num_workers'], shuffle=False)
    analysis(cfg, model, test_loader)

