import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def TrilinearInterpolation(self, img, LUT):
        img = (img - 0.5) * 2.
        img = img.permute(0, 2, 3, 1)[:, None]  # [b, 1, h, w, 3]
        out = F.grid_sample(LUT, img, mode='bilinear', padding_mode='border', align_corners=True)  # [b, 3, 1, h, w]
        out = out[:, :, 0]
        return out


class Three_Dimensional_LUT(BaseModel):
    def __init__(self, cfg):
        super(Three_Dimensional_LUT, self).__init__()
        self.N = cfg.network['n_points']
        self.identity = self.make_identity_LUT(cfg.network['n_points'])

    def make_identity_LUT(self, n_points):
        coords = torch.linspace(0., 1., n_points)
        z, y, x = torch.meshgrid(coords, coords, coords, indexing='ij')
        identity = torch.stack([x, y, z], dim=0)                # [3=(R, G, B), Ns, Ns, Ns]
        return nn.Parameter(identity).requires_grad_(False)
