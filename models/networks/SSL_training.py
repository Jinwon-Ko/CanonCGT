import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import load_pickle
from models.base import Three_Dimensional_LUT
from models.networks.Estimator.Estimator_modules import Embedding_Net, LookUpTable_Estimator


class Net(Three_Dimensional_LUT):
    def __init__(self, cfg):
        super(Net, self).__init__(cfg)

        self.exp_name = cfg.yaml
        self.Embedding_Net = Embedding_Net(cfg)
        self.Canonicalizer = LookUpTable_Estimator(cfg)
        self.Restyler = LookUpTable_Estimator(cfg)

        self.Embedding_Net.freeze_params_()

    def forward(self, inp, ref):
        outs = {}

        # Step 1: Extract style vector
        src_style_vector = self.Embedding_Net(inp)
        tgt_style_vector = self.Embedding_Net(ref)
        outs['src_style_vector'] = src_style_vector

        # Step 2 : Canonicalize
        canonical_out = self.Canonicalizer(inp, condition=src_style_vector)
        outs['canonicalize_LUT'] = canonical_out['LUT']
        outs['canonicalized'] = canonical_out['result']

        # Step 3 : Stylize
        restyler_out = self.Restyler(outs['canonicalized'], condition=tgt_style_vector)
        outs['restylize_LUT'] = restyler_out['LUT']
        outs['restyled'] = restyler_out['result']

        return outs

    def forward_Canonicalizer(self, img):
        # Canonicalizer - Use an input style vector as a condition
        src_style_vector = self.Embedding_Net(img)
        canonical_out = self.Canonicalizer(img, condition=src_style_vector)

        outs = {'src_style_vector': src_style_vector,
                'canonicalize_LUT': canonical_out['LUT'],
                'canonicalized': canonical_out['result']}

        return outs

    def forward_Restyler(self, img, ref):
        # Restyler - Use a reference style vector as a condition
        tgt_style_vector = self.Embedding_Net(ref)
        restyler_out = self.Restyler(img, condition=tgt_style_vector)
        outs = {'tgt_style_vector': tgt_style_vector,
                'restylize_LUT': restyler_out['LUT'],
                'restyled': restyler_out['result']}

        return outs


class CanonCGT_SSL(Three_Dimensional_LUT):
    def __init__(self, cfg):
        super(CanonCGT_SSL, self).__init__(cfg)

        self.Embedding_Net = Embedding_Net(cfg)
        self.Canonicalizer = LookUpTable_Estimator(cfg)
        self.Restyler = LookUpTable_Estimator(cfg)

    def forward(self, inp, ref):
        outs = {}

        # Step 1: Extract style vector
        src_style_vector = self.Embedding_Net(inp)
        tgt_style_vector = self.Embedding_Net(ref)

        # Step 2 : Canonicalize
        canonical_out = self.Canonicalizer(inp, condition=src_style_vector)
        outs['canonicalize_LUT'] = canonical_out['LUT']
        outs['canonicalized'] = canonical_out['result']

        # Step 3 : Stylize
        restyler_out = self.Restyler(outs['canonicalized'], condition=tgt_style_vector)
        outs['restylize_LUT'] = restyler_out['LUT']
        outs['restyled'] = restyler_out['result']

        return outs
