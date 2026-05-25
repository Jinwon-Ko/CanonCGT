import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import resnet18, ResNet18_Weights


# ========================== Attention blocks ==========================
class EncoderBlock(nn.Module):
    def __init__(self, dim, n_block, n_heads, pre_norm=True):
        super(EncoderBlock, self).__init__()
        self.self_attention = nn.ModuleList()
        self.feed_forward = nn.ModuleList()
        for _ in range(n_block):
            self.self_attention.append(Spatial_Self_Attention(dim, n_heads, pre_norm=pre_norm))
            self.feed_forward.append(Point_Feed_Forward_FiLM(dim, pre_norm=pre_norm))

    def forward(self, x, style_vector):
        for self_attention, feed_forward in zip(self.self_attention, self.feed_forward):
            x = x + self_attention(x)
            x = x + feed_forward(x, style_vector)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, n_block, n_heads, pre_norm=True):
        super(DecoderBlock, self).__init__()
        diff_src_kv = 1
        same_src_kv = n_block - 1

        self.cross_attention1 = nn.ModuleList()
        self.feed_forward1 = nn.ModuleList()
        for _ in range(diff_src_kv):
            self.cross_attention1.append(LUT_CA(dim, n_heads, pre_norm=pre_norm))
            self.feed_forward1.append(LUT_FFN_FiLM(dim, pre_norm=pre_norm))

        self.cross_attention2 = nn.ModuleList()
        self.feed_forward2 = nn.ModuleList()
        for _ in range(same_src_kv):
            self.cross_attention2.append(LUT_CA(dim, n_heads, pre_norm=pre_norm))
            self.feed_forward2.append(LUT_FFN_FiLM(dim, pre_norm=pre_norm))

    def forward(self, src_q, src_kv, style_vector):
        for cross_attention, feed_forward in zip(self.cross_attention1, self.feed_forward1):
            src_q = src_q + cross_attention(src_q, src_kv)
            src_q = src_q + feed_forward(src_q, style_vector)

        for cross_attention, feed_forward in zip(self.cross_attention2, self.feed_forward2):
            src_q = src_q + cross_attention(src_q, src_kv)
            src_q = src_q + feed_forward(src_q, style_vector)
        return src_q


class FiLM_Layer(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(FiLM_Layer, self).__init__()

        self.scale = nn.Sequential(nn.Linear(dim, dim),
                                   nn.GELU(),
                                   nn.Linear(dim, hidden_dim))

        self.shift = nn.Sequential(nn.Linear(dim, dim),
                                   nn.GELU(),
                                   nn.Linear(dim, hidden_dim))

    def forward(self, x):
        return self.scale(x), self.shift(x)


# ========================== Encoder blocks ==========================
class Spatial_Self_Attention(nn.Module):
    def __init__(self, dim, n_heads=1, pre_norm=True):
        super(Spatial_Self_Attention, self).__init__()
        assert dim % n_heads == 0

        self.pre_norm = pre_norm
        if pre_norm:
            self.norm = nn.LayerNorm(dim)

        self.n_heads = n_heads
        self.depth = dim // n_heads

        self.tau = nn.Parameter(torch.zeros(n_heads, 1, 1))
        self.qkv_embed = nn.Linear(dim, dim * 3, bias=False)
        self.linear = nn.Linear(dim, dim)

    def forward(self, feat):
        ori_shape = feat.shape
        feat = feat.flatten(2).permute(0, 2, 1)
        if self.pre_norm:
            feat = self.norm(feat)

        # 1. QKV Embed
        qkv = self.qkv_embed(feat)
        q, k, v = qkv.chunk(chunks=3, dim=-1)  # [b, h*w, c]

        # 2. Split heads
        q_feat = self.split_heads(q)  # [b, n, h*w, d]
        k_feat = self.split_heads(k)  # [b, n, h*w, d]
        v_feat = self.split_heads(v)  # [b, n, h*w, d]

        # 3. Self Attention
        self_attn_weight = F.softmax((q_feat @ k_feat.transpose(-2, -1)) * torch.exp(self.tau),
                                     dim=-1)  # [b, n, h*w, h*w]
        res = (self_attn_weight @ v_feat)  # [b, n, h*w, d]
        res = self.merge_heads(res)  # [b, h*w, c]
        res = self.linear(res)  # [b, h*w, c]
        res = res.permute(0, 2, 1).reshape(ori_shape)  # [b, c, h, w]
        return res

    def split_heads(self, x):
        # [b, n_token, c] -> [b, n_heads, n_token, d]
        return x.view(len(x), -1, self.n_heads, self.depth).permute(0, 2, 1, 3)

    def merge_heads(self, x):
        # [b, n_heads, n_token, d] -> [b, n_token, c]
        return x.permute(0, 2, 1, 3).reshape(len(x), -1, self.n_heads * self.depth)


class Point_Feed_Forward_FiLM(nn.Module):
    def __init__(self, dim, pre_norm=True):
        super(Point_Feed_Forward_FiLM, self).__init__()

        self.pre_norm = pre_norm
        if pre_norm:
            self.norm = nn.GroupNorm(1, dim)  # equivalent with LayerNorm

        hidden_dim = dim * 4
        self.ffn1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.ffn2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)

        self.FiLM = FiLM_Layer(dim, hidden_dim)

    def forward(self, feat, style_vector):
        if self.pre_norm:
            feat = self.norm(feat)

        res = self.act(self.ffn1(feat))

        scale, shift = self.FiLM(style_vector)
        scale = scale.view(-1, res.size(1), 1, 1)
        shift = shift.view(-1, res.size(1), 1, 1)
        res = res * scale + shift

        res = self.ffn2(res)
        return res


# ========================== Decoder blocks ==========================
class LUT_CA(nn.Module):
    def __init__(self, dim, n_heads=1, pre_norm=True):
        super(LUT_CA, self).__init__()
        assert dim % n_heads == 0

        self.pre_norm = pre_norm
        if pre_norm:
            self.norm_q = nn.LayerNorm(dim)
            self.norm_kv = nn.LayerNorm(dim)

        self.n_heads = n_heads
        self.depth = dim // n_heads

        self.tau = nn.Parameter(torch.zeros(n_heads, 1, 1))
        self.q_embed = nn.Linear(dim, dim, bias=False)
        self.k_embed = nn.Linear(dim, dim, bias=False)
        self.v_embed = nn.Linear(dim, dim, bias=False)
        self.linear = nn.Linear(dim, dim)

    def forward(self, LUT_token, feat_kv):
        ori_shape = LUT_token.shape
        LUT_token = LUT_token.flatten(2).permute(0, 2, 1)   # [b, M, c]
        feat_kv = feat_kv.flatten(2).permute(0, 2, 1)       # [b, h*w, c]

        if self.pre_norm:
            LUT_token = self.norm_q(LUT_token)
            feat_kv = self.norm_kv(feat_kv)

        # 1. QKV Embed
        q_LUT = self.q_embed(LUT_token)     # [b, M, c]
        k_feat = self.k_embed(feat_kv)      # [b, h*w, c]
        v_feat = self.v_embed(feat_kv)      # [b, h*w, c]
        q_LUT = self.split_heads(q_LUT)     # [b, nh, M, d]
        k_feat = self.split_heads(k_feat)   # [b, nh, h*w, d]
        v_feat = self.split_heads(v_feat)   # [b, nh, h*w, d]

        # 2. Cross Attention
        cross_attn_weight = F.softmax((q_LUT @ k_feat.transpose(-2, -1)) * torch.exp(self.tau), dim=-1)  # [b, nh, M, h*w]

        res = (cross_attn_weight @ v_feat)  # [b, nh, M, d]
        res = self.merge_heads(res)  # [b, M, c]
        res = self.linear(res)
        res = res.permute(0, 2, 1).reshape(ori_shape)
        return res

    def split_heads(self, x):
        # [b, n_token, c] -> [b, n_heads, n_token, d]
        return x.view(len(x), -1, self.n_heads, self.depth).permute(0, 2, 1, 3)

    def merge_heads(self, x):
        # [b, n_heads, n_token, d] -> [b, n_token, c]
        return x.permute(0, 2, 1, 3).reshape(len(x), -1, self.n_heads * self.depth)


class LUT_FFN_FiLM(nn.Module):
    def __init__(self, dim, pre_norm=True):
        super(LUT_FFN_FiLM, self).__init__()
        self.pre_norm = pre_norm
        if pre_norm:
            self.norm = nn.GroupNorm(1, dim)  # equivalent with LayerNorm

        hidden_dim = dim * 4
        self.ffn1 = nn.Conv3d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.ffn2 = nn.Conv3d(hidden_dim, dim, kernel_size=1)

        self.FiLM = FiLM_Layer(dim, hidden_dim)

    def forward(self, LUT_token, style_vector):
        if self.pre_norm:
            LUT_token = self.norm(LUT_token)

        res = self.act(self.ffn1(LUT_token))

        scale, shift = self.FiLM(style_vector)
        scale = scale.view(-1, res.size(1), 1, 1, 1)
        shift = shift.view(-1, res.size(1), 1, 1, 1)
        res = scale * res + shift

        res = self.ffn2(res)
        return res


# ========================== Downsample methods ==========================
class MobileNetv2_based_backbone(nn.Module):
    def __init__(self, dim):
        super(MobileNetv2_based_backbone, self).__init__()

        weights = MobileNet_V2_Weights.DEFAULT
        mobilenet = mobilenet_v2(weights=weights)
        self.backbone = nn.Sequential(*list(mobilenet.features)[:7])      # [32, H//8, W//8]
        backbone_dim = 32

        self.channel_fitting = nn.Sequential(
            nn.Conv2d(backbone_dim, dim, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        return self.channel_fitting(self.backbone(x))


class MobileNetv2_based_style_encoder(nn.Module):
    def __init__(self, dim):
        super(MobileNetv2_based_style_encoder, self).__init__()

        weights = MobileNet_V2_Weights.DEFAULT
        mobilenet = mobilenet_v2(weights=weights)
        self.backbone = nn.Sequential(*list(mobilenet.features))      # [H//32, W//32]
        backbone_dim = mobilenet.last_channel

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_fitting = nn.Sequential(
            nn.Linear(backbone_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim))

    def forward(self, x):
        cls_embed = self.avg_pool(self.backbone(x))
        cls_embed = cls_embed.view(len(x), -1)
        return self.channel_fitting(cls_embed)
