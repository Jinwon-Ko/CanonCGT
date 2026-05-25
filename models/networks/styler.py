import torch
import torch.nn as nn

from utils.util import load_pickle
from models.base import Three_Dimensional_LUT
from models.networks.Estimator.Estimator_modules import LookUpTable_Estimator


class Net(Three_Dimensional_LUT):
    def __init__(self, cfg):
        super(Net, self).__init__(cfg)

        # self.Embedding_Net = Embedding_Net(cfg)
        # self.Canonicalizer = LookUpTable_Estimator(cfg)
        self.Restyler = LookUpTable_Estimator(cfg)

        centroids = load_pickle(cfg.checkpoint['centroids'])
        style_centroids = {f'{k:02d}': nn.Parameter(torch.tensor(v, dtype=torch.float32)) for k, v in centroids.items()}
        self.style_centroids = nn.ParameterDict(style_centroids)
        for p in self.style_centroids.parameters():
            p.requires_grad_(False)

    def forward(self, img, style_idx):
        condition = torch.stack([self.style_centroids[idx] for idx in style_idx])
        condition = condition.to(img.device)

        outs = {}
        restyler_out = self.Restyler(img, condition)
        outs['restyled'] = restyler_out['result']
        return outs
