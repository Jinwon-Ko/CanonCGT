import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from utils.util import *


class Loss_Function(nn.Module):
    def __init__(self, cfg, weight_dict):
        super().__init__()
        self.cfg = cfg
        self.weight_dict = weight_dict

        self.loss_SmoothL1 = nn.SmoothL1Loss(beta=0.5)
        self.loss_L1 = nn.L1Loss()
        self.loss_L2 = nn.MSELoss()
        self.relu = nn.ReLU()

        perceptual_backbone = models.vgg16(pretrained=True)
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]), requires_grad=False)
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]), requires_grad=False)

        blocks = []
        blocks.append(perceptual_backbone.features[:4].eval())
        blocks.append(perceptual_backbone.features[4:9].eval())
        blocks.append(perceptual_backbone.features[9:16].eval())
        blocks.append(perceptual_backbone.features[16:23].eval())
        blocks.append(perceptual_backbone.features[23:30].eval())
        self.blocks = nn.ModuleList(blocks)
        for name, p in self.named_parameters():
            p.requires_grad = False
            
    def get_SupCon_loss(self, features, labels, anchor_mask=None, temperature=0.05, l_name='SupCon'):
        labels = labels.contiguous().view(-1, 1)  # [b, 1]
        mask = torch.eq(labels, labels.T).float().to(features.device)  # [b, b]

        similarity_matrix = torch.matmul(features, features.T) / temperature  # [b, b]

        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # remove self-comparisons
        eye = torch.eye(mask.shape[0], dtype=torch.bool).to(features.device)
        logits.masked_fill_(eye, -1e9)
        mask.masked_fill_(eye, 0)

        # log_prob: [B, B]
        log_prob = F.log_softmax(logits, dim=1)
        if anchor_mask is not None:     # for Stage 2
            per_row = -(mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
            loss = per_row[anchor_mask]
        else:                           # for Stage 1
            loss = -(mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        return {l_name: self.weight_dict['loss_SupCon'] * loss.mean()}

    def get_SupCon_positive_loss(self, features, l_name='SupConPos'):
        """
        Positive-only loss (all samples are considered positives).
        Features: [B, D], normalized embeddings recommended.
        """
        device = features.device
        B = features.size(0)

        # Normalize features (cosine similarity)
        cos = torch.matmul(features, features.T)  # [B, B]
        cos = torch.clamp(cos, -1.0, 1.0)

        # Self removal
        eye = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = (~eye).float()  # 1 for off-diagonal, 0 for diagonal

        # Positive-only distance (1 - cos)
        pos_dist = (1.0 - cos) * pos_mask

        # Average per anchor
        per_anchor = pos_dist.sum(1) / (pos_mask.sum(1) + 1e-6)
        return {l_name: self.weight_dict['loss_SupCon'] * per_anchor.mean()}

    def get_Center_loss(self, centroids, feats, l_name='center'):
        cosine_sim = (feats * centroids).sum(-1)
        loss = (1 - cosine_sim)
        return {l_name: self.weight_dict['loss_center'] * loss.mean()}

    def get_canonicalizer_loss(self, pred, gt, pair_types=None, l_name='canonical'):
        if pair_types is not None:      # for Stage 3
            kk_pair_indices = [idx for idx, pair_type in enumerate(pair_types) if pair_type == 'kk']
        else:                           # for Stage 1, 2
            kk_pair_indices = list(range(len(pred)))

        losses = {}
        losses.update(self.get_reconsturction_loss(pred[kk_pair_indices], gt[kk_pair_indices], l_name='recon'))
        losses.update(self.get_grad_loss(pred[kk_pair_indices], gt[kk_pair_indices], l_name='grad'))
        losses.update(self.get_perceptual_loss(pred[kk_pair_indices], gt[kk_pair_indices], l_name='perceptual'))

        return {l_name: self.weight_dict['loss_canonical'] * sum(losses[k] for k in losses.keys())}

    def get_restyler_loss(self, pred, gt, l_name='restyler'):
        losses = {}
        losses.update(self.get_reconsturction_loss(pred, gt, l_name='recon'))
        losses.update(self.get_grad_loss(pred, gt, l_name='grad'))
        losses.update(self.get_perceptual_loss(pred, gt, l_name='perceptual'))
        return {l_name: self.weight_dict['loss_styler'] * sum(losses[k] for k in losses.keys())}

    def get_reconsturction_loss(self, pred, gt, l_name='recon'):
        return {l_name: self.weight_dict['loss_recon'] * self.loss_L2(pred, gt)}

    def get_grad_loss(self, pred, gt, l_name='grad'):
        grad_y_pred = (pred[:, :, 1:, :] - pred[:, :, :-1, :])      # H
        grad_x_pred = (pred[:, :, :, 1:] - pred[:, :, :, :-1])      # W

        grad_y_gt = (gt[:, :, 1:, :] - gt[:, :, :-1, :])            # H
        grad_x_gt = (gt[:, :, :, 1:] - gt[:, :, :, :-1])            # W

        grad_y = self.loss_L2(grad_y_pred, grad_y_gt)               # H
        grad_x = self.loss_L2(grad_x_pred, grad_x_gt)               # W
        return {l_name: self.weight_dict['loss_recon'] * (grad_y + grad_x)}

    def get_perceptual_loss(self, pred, gt, feature_layers=[0, 1, 2, 3], l_name='perceptual'):
        x = (pred - self.mean.view(1, -1, 1, 1)) / self.std.view(1, -1, 1, 1)
        y = (gt - self.mean.view(1, -1, 1, 1)) / self.std.view(1, -1, 1, 1)

        loss = 0.0
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += self.loss_L1(x, y)
        return {l_name: self.weight_dict['loss_perceptual'] * loss / len(pred)}




def get_loss_function(cfg):
    weight_dict = dict()
    if 'style_encoder' in cfg.yaml:
        weight_dict['loss_SupCon'] = cfg.training['loss_SupCon_coef']

    if 'canonicalizer' in cfg.yaml:
        weight_dict['loss_recon'] = cfg.training['loss_recon_coef']
        weight_dict['loss_perceptual'] = cfg.training['loss_perceptual_coef']

    if 'styler' in cfg.yaml:
        weight_dict['loss_recon'] = cfg.training['loss_recon_coef']
        weight_dict['loss_perceptual'] = cfg.training['loss_perceptual_coef']

    if 'end_to_end_finetuning' in cfg.yaml:
        weight_dict['loss_SupCon'] = cfg.training['loss_SupCon_coef']
        weight_dict['loss_recon'] = cfg.training['loss_recon_coef']
        weight_dict['loss_perceptual'] = cfg.training['loss_perceptual_coef']
        weight_dict['loss_canonical'] = cfg.training['loss_canonical_coef']
        weight_dict['loss_styler'] = cfg.training['loss_styler_coef']

    if 'SSL_training' in cfg.yaml:
        weight_dict['loss_recon'] = cfg.training['loss_recon_coef']
        weight_dict['loss_perceptual'] = cfg.training['loss_perceptual_coef']
        weight_dict['loss_canonical'] = cfg.training['loss_canonical_coef']
        weight_dict['loss_styler'] = cfg.training['loss_styler_coef']

    criterion = Loss_Function(cfg, weight_dict=weight_dict)
    return criterion
