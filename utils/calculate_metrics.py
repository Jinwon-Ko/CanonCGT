import cv2
import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

from utils.util import to_np, to_tensor
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from pretrained.edge_detector_LDC.modelB4 import LDC



class Evaluator(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.lpips_tool = lpips.LPIPS(net='vgg')
        self.lpips_tool.eval()
        # self.edge_tool = EdgeExtractor_LDC(device='cuda')

    def make_data2img(self, data):
        data = to_np(data.permute(1, 2, 0))
        data = np.clip(data, a_min=0.0, a_max=1.0)
        data = (data * 255).astype(np.uint8)[:, :, [2, 1, 0]]
        return data

    # --------------------------------------------------------
    # Fidelity metrics
    # PSNR
    @torch.no_grad()
    def measure_PSNR(self, pred, gt, reduction='mean'):
        assert len(pred) == len(gt)

        PSNR_list = []
        for b_idx in range(len(pred)):
            image_true = self.make_data2img(gt[b_idx])
            image_test = self.make_data2img(pred[b_idx])
            score = psnr(image_true, image_test, data_range=255)
            if np.isinf(score):
                score = 50.0
            PSNR_list.append(float(score))

        if reduction == 'sum':
            return sum(PSNR_list)
        elif reduction == 'mean':
            return sum(PSNR_list) / len(PSNR_list)
        elif reduction is None:
            return PSNR_list

    # SSIM
    @torch.no_grad()
    def measure_SSIM(self, inp, out, reduction='mean'):
        assert len(inp) == len(out)

        SSIM_list = []
        for b_idx in range(len(inp)):
            img1 = to_np(inp[b_idx].permute(1, 2, 0))
            img2 = to_np(out[b_idx].permute(1, 2, 0))
            score = ssim(img1, img2, channel_axis=2, data_range=1.0)
            SSIM_list.append(float(score))

        if reduction == 'sum':
            return sum(SSIM_list)
        elif reduction == 'mean':
            return sum(SSIM_list) / len(SSIM_list)
        elif reduction is None:
            return SSIM_list

    @torch.no_grad()
    def measure_DeltaEab(self, pred, gt):
        return deltaE_ab(pred, gt)      # deltaE_2000(pred, gt)        # deltaE_ab(pred, gt)

    # ---------------------------
    # Structure Preservation metrics
    # ---------------------------
    @torch.no_grad()
    def measure_LPIPS(self, pred, gt, reduction='mean'):
        # LPIPS expects input in [-1,1]
        pred_scaled = pred * 2 - 1
        gt_scaled = gt * 2 - 1
        dist = self.lpips_tool(pred_scaled, gt_scaled)
        return float((dist.mean() if reduction == 'mean' else dist.sum()).item())

    @torch.no_grad()
    def measure_SSIM_ED(self, pred, inp, reduction='mean'):
        edge_pred = self.edge_tool.get_edge(pred)
        edge_inp = self.edge_tool.get_edge(inp)
        return self.measure_SSIM(edge_pred, edge_inp, reduction=reduction)

    # ---------------------------
    # Style metrics
    # ---------------------------
    @torch.no_grad()
    def measure_HCorr(self, pred, ref, reduction='mean', bins=256):
        B, C, H, W = pred.shape
        vals = []
        for b in range(B):
            corr_per_ch = []
            for c in range(C):
                ref_ch = ref[b, c].clamp(0, 1)
                out_ch = pred[b, c].clamp(0, 1)

                ref_hist = torch.histc(ref_ch, bins=bins, min=0, max=1)
                out_hist = torch.histc(out_ch, bins=bins, min=0, max=1)

                ref_hist = ref_hist / ref_hist.sum()
                out_hist = out_hist / out_hist.sum()

                ref_mean = ref_hist.mean()
                out_mean = out_hist.mean()

                num = ((ref_hist - ref_mean) * (out_hist - out_mean)).sum()
                den = torch.sqrt(((ref_hist - ref_mean) ** 2).sum()) * torch.sqrt(((out_hist - out_mean) ** 2).sum())
                corr = num / (den + 1e-8)
                corr_per_ch.append(corr)
            vals.append(torch.stack(corr_per_ch).mean())    # mean over 3 channels
        vals = torch.stack(vals)
        return float(vals.mean().item() if reduction == 'mean' else vals.sum().item())

    @torch.no_grad()
    def measure_HChi(self, pred, ref, reduction='mean', bins=256):
        B, C, H, W = pred.shape
        vals = []
        eps = 1e-6
        for b in range(B):
            chi_per_ch = []
            for c in range(C):
                ref_ch = ref[b, c].clamp(0, 1)
                out_ch = pred[b, c].clamp(0, 1)

                ref_hist = torch.histc(ref_ch, bins=bins, min=0, max=1)
                out_hist = torch.histc(out_ch, bins=bins, min=0, max=1)

                ref_hist = ref_hist / ref_hist.sum()
                out_hist = out_hist / out_hist.sum()

                chi = 0.5 * torch.sum(((ref_hist - out_hist) ** 2) / (ref_hist + out_hist + eps))
                chi_per_ch.append(chi)
            vals.append(torch.stack(chi_per_ch).mean())     # mean over 3 channels
        vals = torch.stack(vals)
        return float(vals.mean().item() if reduction == 'mean' else vals.sum().item())



class Evaluator_SupCon(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.knn = KNeighborsClassifier(n_neighbors=20)

    def measure_knn_accuracy(self, feats, labels, return_pred=False):
        feats = to_np(feats)
        labels = to_np(labels)

        self.knn.fit(feats, labels)
        pred = self.knn.predict(feats)
        acc = accuracy_score(labels, pred)

        if return_pred:
            return acc, pred
        else:
            return acc


# --------------------------------------------------------
# Histogram-based metrics
# --------------------------------------------------------
def calc_histogram(img, bins=256, hist_range=(0, 256)):
    """
    Compute normalized histogram for each channel.
    Input: img (np.uint8, shape [H,W,3])
    """
    hist_list = []
    for ch in range(3):
        hist = cv2.calcHist([img], [ch], None, [bins], hist_range)
        # hist = cv2.normalize(hist, hist).flatten()
        hist = cv2.normalize(hist, None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1).ravel()
        hist_list.append(hist)
    return hist_list

def hist_correlation(ref_img, out_img):
    """
    Histogram Correlation (H-Corr): higher is better
    """
    ref_hists = calc_histogram(ref_img)
    out_hists = calc_histogram(out_img)
    corr_list = [cv2.compareHist(ref_hists[i], out_hists[i], cv2.HISTCMP_CORREL) for i in range(3)]
    return float(np.mean(corr_list))

def hist_chi_square(ref_img, out_img):
    """
    Histogram Chi-squared distance (H-Chi): lower is better
    """
    ref_hists = calc_histogram(ref_img)
    out_hists = calc_histogram(out_img)
    chi_list = [cv2.compareHist(ref_hists[i], out_hists[i], cv2.HISTCMP_CHISQR) for i in range(3)]
    return float(np.mean(chi_list))

# --------------------------------------------------------
# Gram metric (already compatible with your VGG_Style)
# --------------------------------------------------------
class GramMetric(nn.Module):
    def __init__(self, layers=('0','5','10','19','28')):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        self.layers = layers
        self.vgg = vgg
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    def get_features(self, x):
        feats = {}
        x = self.norm(x)
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                feats[name] = x
        return feats

    def gram_matrix(self, feat):
        B, C, H, W = feat.shape
        F = feat.view(B, C, H*W)
        G = torch.bmm(F, F.transpose(1,2))
        return G

    @torch.no_grad()
    def compute(self, ref, out):
        f_ref = self.get_features(ref)
        f_out = self.get_features(out)
        loss = 0
        for l in self.layers:
            B, C, H, W = f_ref[l].shape
            G_ref = self.gram_matrix(f_ref[l])
            G_out = self.gram_matrix(f_out[l])
            loss += F.mse_loss(G_ref, G_out) / (C * H * W) ** 2
        return loss.item()


# --------------------------------------------------------
# LPIPS metric
# --------------------------------------------------------
class LPIPSMetric(nn.Module):
    def __init__(self, net='vgg'):
        super().__init__()
        # net: ['alex','vgg','squeeze']
        self.loss_fn = lpips.LPIPS(net=net)

    @torch.no_grad()
    def compute(self, img1, img2):
        # img1,img2: torch.Tensor, [1,3,H,W], range [-1,1]
        dist = self.loss_fn(img1, img2)
        return dist.item()


# ========================================================
# Utility: RGB -> Lab + ΔEab
# ========================================================
def rgb_to_lab(rgb):
    """
    Convert RGB [0,1] tensor to Lab (D65, sRGB)
    rgb: [B,3,H,W]
    """
    rgb = rgb.permute(0, 2, 3, 1)
    mask = rgb > 0.04045
    rgb = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    rgb = rgb * 100

    xyz_mat = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], device=rgb.device, dtype=rgb.dtype)
    xyz = torch.matmul(rgb, xyz_mat.T)
    xyz_ref = torch.tensor([95.047, 100.000, 108.883], device=rgb.device, dtype=rgb.dtype)
    xyz = xyz / xyz_ref

    eps, kappa = 0.008856, 903.3
    mask = xyz > eps
    f_xyz = torch.where(mask, xyz ** (1 / 3), (kappa * xyz + 16) / 116)

    L = 116 * f_xyz[..., 1] - 16
    a = 500 * (f_xyz[..., 0] - f_xyz[..., 1])
    b = 200 * (f_xyz[..., 1] - f_xyz[..., 2])

    lab = torch.stack([L, a, b], dim=-1)
    return lab.permute(0, 3, 1, 2)  # [B,3,H,W]


@torch.no_grad()
def deltaE_ab(img1, img2):
    """Mean ΔEab (CIE76) between two RGB [0,1] tensors."""
    lab1 = rgb_to_lab(img1)
    lab2 = rgb_to_lab(img2)
    diff = lab1 - lab2
    de = torch.sqrt((diff ** 2).sum(dim=1))
    return de.mean().item()


# --------------------------------------------------------
# Edge detector (LDC customized)
# --------------------------------------------------------
class EdgeExtractor_LDC:
    def __init__(self,
                 weight_path='./pretrained/LDC_pretrained_model.pth',
                 mean_bgr=[103.939, 116.779, 123.68],
                 device='cuda',
                 long_side=2048):

        self.device = device
        self.mean_bgr = torch.tensor(mean_bgr, dtype=torch.float32).view(1, 3, 1, 1).to(device)
        self.model = LDC().to(device)
        ckpt = torch.load(weight_path, map_location=device)
        self.model.load_state_dict(ckpt)
        self.model.eval()
        self.long_side = long_side

    @torch.no_grad()
    def get_edge(self, img_tensor):
        """
        img_tensor: torch.Tensor [B,3,H,W], RGB, range [0,1]
        return: edge map tensor [B,1,H,W], range [0,1]
        """
        assert img_tensor.ndim == 4 and img_tensor.shape[1] == 3, \
            "Input must be [B,3,H,W] RGB normalized tensor"

        B, C, H, W = img_tensor.shape

        # 1. RGB → BGR
        img_bgr = img_tensor[:, [2, 1, 0], :, :] * 255.0

        # 2. mean subtraction (BGR sequence)
        img_bgr = img_bgr - self.mean_bgr

        # 3. resize to 512 x 512 to align pretrained-model settings
        img_bgr = F.interpolate(img_bgr, size=(512, 512), mode='bilinear', align_corners=False)

        # 4. Forward through LDC
        preds = self.model(img_bgr)
        if isinstance(preds, (list, tuple)):
            preds = preds[-1]  # final edge map

        # 5. Sigmoid activation
        edge = torch.sigmoid(preds)

        # # 6. resize to long_side=2048 and enforce divisibility by 8 (keeping aspect)
        # scale = self.long_side / max(H, W)
        # new_h = int(round(H * scale))
        # new_w = int(round(W * scale))
        # target_h = int(np.ceil(new_h / 8) * 8)
        # target_w = int(np.ceil(new_w / 8) * 8)
        # edge = F.interpolate(edge, size=(target_h, target_w), mode='bilinear', align_corners=False)

        return edge.clamp(0, 1)
