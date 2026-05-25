from models.base import Three_Dimensional_LUT
from models.networks.Estimator.Estimator_blocks import *


# ========================== Networks ==========================
class Embedding_Net(nn.Module):
    def __init__(self, cfg):
        super(Embedding_Net, self).__init__()

        dim = cfg.network['style_token_dim']
        self.downsample = nn.Upsample(size=(224, 224), mode='bilinear')
        self.net = MobileNetv2_based_style_encoder(dim)

    def freeze_params_(self):
        for name, p in self.named_parameters():
            p.requires_grad = False

    def forward(self, x):
        latent = self.net(self.downsample(x))
        style_token = F.normalize(latent, dim=-1)
        return style_token


class LookUpTable_Estimator(Three_Dimensional_LUT):
    def __init__(self, cfg):
        super(LookUpTable_Estimator, self).__init__(cfg)

        self.downsample = nn.Upsample(size=(224, 224), mode='bilinear')
        self.feature_extractor = Feature_extractor(cfg)
        self.generate_lut = LUT_token_update(cfg)

    def freeze_params_(self):
        for name, p in self.named_parameters():
            p.requires_grad = False

    def forward(self, img, condition):
        resized_img = self.downsample(img)                              # [b, 3, h, w]
        image_tokens = self.feature_extractor(resized_img, condition)   # [b, c, h//8, w//8]

        identity = self.identity[None].repeat(len(img), 1, 1, 1, 1)     # [b, 3, N, N, N]
        LUT = self.generate_lut(identity, image_tokens, condition)      # [b, 3, N, N, N]

        result = self.TrilinearInterpolation(img, LUT)
        return {'LUT': LUT, 'result': result}


# ========================== Encoder ==========================
class Feature_extractor(nn.Module):
    def __init__(self, cfg):
        super(Feature_extractor, self).__init__()

        dim = cfg.network['hidden_dim']
        self.down_block = MobileNetv2_based_backbone(dim)

        self.enc_block = nn.ModuleList()
        for level in range(cfg.network['n_attn_module']):
            n_enc_block = cfg.network['n_enc_block'][level]
            n_enc_heads = cfg.network['n_enc_heads'][level]
            self.enc_block.append(EncoderBlock(dim, n_enc_block, n_enc_heads, pre_norm=True))

    def forward(self, content_img, style_token):
        feat = self.down_block(content_img)
        pixel_tokens = [feat]

        for enc_block in self.enc_block:
            feat = enc_block(feat, style_token)
            pixel_tokens.append(feat)

        return pixel_tokens


# ========================== Decoder ==========================
class LUT_token_update(nn.Module):
    def __init__(self, cfg):
        super(LUT_token_update, self).__init__()

        dim = cfg.network['hidden_dim']

        self.start_block = DecoderBlock(dim, n_block=cfg.network['n_dec_block'][0], n_heads=cfg.network['n_dec_heads'][0], pre_norm=True)
        self.dec_block = nn.ModuleList()
        for level in range(cfg.network['n_attn_module']):
            n_dec_block = cfg.network['n_dec_block'][level]
            n_dec_heads = cfg.network['n_dec_heads'][level]
            self.dec_block.append(DecoderBlock(dim, n_dec_block, n_dec_heads, pre_norm=True))

        self.inc = nn.Sequential(nn.Conv3d(3, dim, kernel_size=1),
                                 nn.GELU(),
                                 nn.Conv3d(dim, dim, kernel_size=1))
        self.outc = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=1),
                                  nn.GELU(),
                                  nn.Conv3d(dim, 3, kernel_size=1))

    def forward(self, LUT_identity, image_tokens, style_token):
        LUT_token = self.inc(LUT_identity)
        LUT_token = self.start_block(LUT_token, image_tokens[0], style_token)
        for dec_block, token_kv in zip(self.dec_block, image_tokens[1:]):
            LUT_token = dec_block(LUT_token, token_kv, style_token)
        LUT = self.outc(LUT_token) + LUT_identity
        return LUT
