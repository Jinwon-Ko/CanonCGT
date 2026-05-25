import torch

from torch.utils.data import DataLoader
from dataloaders.factory import load_dataset
from engines.end_to_end_finetuning.forward_test import analysis


def test_process(cfg, model):
    if cfg.load:
        checkpoint = torch.load(cfg.checkpoint['model'])
        model.load_state_dict(checkpoint['model_state_dict'])
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

    test_dataset = load_dataset(cfg, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=20, num_workers=cfg.dataset['num_workers'], shuffle=False)
    analysis(cfg, model, test_loader)

