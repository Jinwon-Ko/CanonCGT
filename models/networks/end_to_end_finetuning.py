import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import load_pickle
from models.base import Three_Dimensional_LUT
from models.networks.Estimator.Estimator_modules import Embedding_Net, LookUpTable_Estimator


class Net(Three_Dimensional_LUT):
    def __init__(self, cfg):
        super(Net, self).__init__(cfg)

        self.Embedding_Net = Embedding_Net(cfg)
        self.Canonicalizer = LookUpTable_Estimator(cfg)
        self.Restyler = LookUpTable_Estimator(cfg)

        centroids = load_pickle(cfg.checkpoint['centroids'])
        style_centroids = {f'{k:02d}': nn.Parameter(torch.tensor(v, dtype=torch.float32)) for k, v in centroids.items()}
        self.style_centroids = nn.ParameterDict(style_centroids)
        for p in self.style_centroids.parameters():
            p.requires_grad = False

    def forward(self, img, ref):
        outs = {}

        # Step 1: Extract style vector
        src_style_vector = self.Embedding_Net(img)
        tgt_style_vector = self.Embedding_Net(ref)
        outs['src_style_vector'] = src_style_vector

        # Step 2 : Canonicalize
        canonical_out = self.Canonicalizer(img, condition=src_style_vector)
        outs['canonicalize_LUT'] = canonical_out['LUT']
        outs['canonicalized'] = canonical_out['result']

        # Step 3 : Stylize
        restyler_out = self.Restyler(outs['canonicalized'], condition=tgt_style_vector)
        outs['restylize_LUT'] = restyler_out['LUT']
        outs['restyled'] = restyler_out['result']

        return outs

    def get_style_centroids(self):
        device = self.style_centroids['01'].device
        all_style_labels = torch.tensor([int(k) for k in self.style_centroids.keys()], dtype=torch.long).to(device)
        all_style_centroids = torch.stack([self.style_centroids[k] for k in self.style_centroids.keys()], dim=0)
        all_style_centroids = F.normalize(all_style_centroids, dim=-1)
        return {'all_style_centroids': all_style_centroids,
                'all_style_labels': all_style_labels}

    def forward_Canonicalizer(self, img):
        # Canonicalizer - Use an input style vector as a condition
        src_style_vector = self.Embedding_Net(img)
        canonical_out = self.Canonicalizer(img, condition=src_style_vector)
        return {'src_style_vector': src_style_vector,
                'canonicalize_LUT': canonical_out['LUT'],       # [B, 3, N, N, N]
                'canonicalized': canonical_out['result']}       # [B, 3, H, W]

    def forward_Restyler(self, img, ref):
        # Restyler - Use a reference style vector as a condition
        tgt_style_vector = self.Embedding_Net(ref)
        restyler_out = self.Restyler(img, condition=tgt_style_vector)
        return {'tgt_style_vector': tgt_style_vector,
                'restylize_LUT': restyler_out['LUT'],       # [B, 3, N, N, N]
                'restyled': restyler_out['result']}         # [B, 3, H, W]


class CanonCGT_E2E(Three_Dimensional_LUT):
    def __init__(self, cfg):
        super(CanonCGT_E2E, self).__init__(cfg)

        self.Embedding_Net = Embedding_Net(cfg)
        self.Canonicalizer = LookUpTable_Estimator(cfg)
        self.Restyler = LookUpTable_Estimator(cfg)

    def forward(self, img, ref):
        outs = {}

        # Step 1: Extract style vector
        src_style_vector = self.Embedding_Net(img)
        tgt_style_vector = self.Embedding_Net(ref)
        outs['src_style_vector'] = src_style_vector

        # Step 2 : Canonicalize
        canonical_out = self.Canonicalizer(img, condition=src_style_vector)
        outs['canonicalize_LUT'] = canonical_out['LUT']
        outs['canonicalized'] = canonical_out['result']

        # Step 3 : Stylize
        restyler_out = self.Restyler(outs['canonicalized'], condition=tgt_style_vector)
        outs['restylize_LUT'] = restyler_out['LUT']
        outs['restyled'] = restyler_out['result']

        return outs
