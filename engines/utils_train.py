import os
import torch
import matplotlib.pyplot as plt


# Training helper
def define_losses_and_metrics(cfg):
    if 'style_encoder' in cfg.yaml:
        Losses = {'Total': [], 'SupCon': []}
        Performances = {'acc': []}
        best = {'acc': 0}

    if 'canonicalizer' in cfg.yaml:
        Losses = {'Total': [], 'recon': [], 'grad': [], 'perceptual': []}
        Performances = {'PSNR': []}
        best = {'PSNR': 0}

    if 'styler' in cfg.yaml:
        Losses = {'Total': [], 'recon': [], 'grad': [], 'perceptual': []}
        Performances = {'PSNR': []}
        best = {'PSNR': 0}

    if 'end_to_end_finetuning' in cfg.yaml:
        Losses = {'Total': [], 'canonicalizer': [], 'restyler': [], 'SupCon': []}
        Performances = {'PSNR': []}
        best = {'PSNR': 0}

    if 'SSL_training' in cfg.yaml:
        Losses = {'Total': [], 'canonicalizer': [], 'restyler': []}
        Performances = {'PSNR': []}
        best = {'PSNR': 0}

    now = best
    return Losses, Performances, best, now


def get_loss_names(cfg):
    if 'style_encoder' in cfg.yaml:
        loss_names = ['SupCon']

    if 'canonicalizer' in cfg.yaml:
        loss_names = ['recon', 'grad', 'perceptual']

    if 'styler' in cfg.yaml:
        loss_names = ['recon', 'grad', 'perceptual']

    if 'end_to_end_finetuning' in cfg.yaml:
        loss_names = ['canonicalizer', 'restyler', 'SupCon']

    if 'SSL_training' in cfg.yaml:
        loss_names = ['canonicalizer', 'restyler']

    loss_t = {'Total': 0}
    loss_t.update({key: 0 for key in loss_names})
    return loss_t



def get_total_loss(cfg, criterion, batch, outputs):
    loss_dict = {}
    if 'style_encoder' in cfg.yaml:
        loss_dict.update(criterion.get_SupCon_loss(outputs['style_vector'], batch['style_label'], l_name='SupCon'))

    if 'canonicalizer' in cfg.yaml:
        loss_dict.update(criterion.get_reconsturction_loss(outputs['canonicalized'], batch['gt'], l_name='recon'))
        loss_dict.update(criterion.get_grad_loss(outputs['canonicalized'], batch['gt'], l_name='grad'))
        loss_dict.update(criterion.get_perceptual_loss(outputs['canonicalized'], batch['gt'], l_name='perceptual'))

    if 'styler' in cfg.yaml:
        loss_dict.update(criterion.get_reconsturction_loss(outputs['restyled'], batch['gt'], l_name='recon'))
        loss_dict.update(criterion.get_grad_loss(outputs['restyled'], batch['gt'], l_name='grad'))
        loss_dict.update(criterion.get_perceptual_loss(outputs['restyled'], batch['gt'], l_name='perceptual'))

    if 'end_to_end_finetuning' in cfg.yaml:
        features = torch.cat([outputs['src_style_vector'], outputs['tgt_style_vector'], outputs['all_style_centroids']], dim=0)
        labels = torch.cat([batch['src_style_label'], batch['tgt_style_label'], outputs['all_style_labels']], dim=0)
        anchor_mask = torch.zeros(len(features), device=features.device, dtype=torch.bool)
        anchor_mask[:len(outputs['src_style_vector']) + len(outputs['tgt_style_vector'])] = True

        loss_dict.update(criterion.get_canonicalizer_loss(outputs['canonicalized'], batch['canonicalizer_GT'], l_name='canonicalizer'))
        loss_dict.update(criterion.get_restyler_loss(outputs['restyled'], batch['restyler_GT'], l_name='restyler'))
        loss_dict.update(criterion.get_SupCon_loss(features, labels, anchor_mask, l_name='SupCon'))

    if 'SSL_training' in cfg.yaml:
        Y = outputs['canonicalized']
        Y_gt = batch['canonicalizer_GT']
        Z = outputs['restyled']
        Z_gt = batch['restyler_GT']
        pair_type = batch['pair_type']

        loss_dict.update(criterion.get_canonicalizer_loss(Y, Y_gt, pair_type, l_name='canonicalizer'))
        loss_dict.update(criterion.get_restyler_loss(Z, Z_gt, l_name='restyler'))

    return loss_dict


# Logging
def update_dict(Losses, losses, Performances, performances):
    for key, value in losses.items():
        try:
            Losses[key].append(value.item())
        except:
            Losses[key].append(value)
    for key, value in performances.items():
        Performances[key].append(value)
    return Losses, Performances


def update_LRs(LRs, lr):
    LRs.append(lr)
    return LRs


def save_losses_and_performances(cfg, losses, performances, LRs):
    # Draw Losses
    fig = plt.figure()
    for key, value in losses.items():
        try:
            plt.plot(value.item(), label=f'Loss_{key}')
        except:
            plt.plot(value, label=f'Loss_{key}')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.legend(loc='upper right')

    plt.savefig(os.path.join(cfg.viz_dir, 'Losses.jpg'))
    plt.close()

    # Draw Performances
    fig = plt.figure()
    plt.plot(performances, label='Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')

    plt.savefig(os.path.join(cfg.viz_dir, 'Accuracies.jpg'))
    plt.close()

    # Draw Learning rates
    fig = plt.figure()
    plt.plot(LRs, label='LR')
    plt.xlabel('Epochs')
    plt.ylabel('Learning rates')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(cfg.viz_dir, 'Learning_rate.jpg'))
    plt.close()


def save_learning_rates(cfg, LRs):
    fig = plt.figure()
    plt.plot(LRs, label='LR')
    plt.xlabel('Epochs')
    plt.ylabel('Learning rates')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(cfg.viz_dir, 'Learning_rate.jpg'))
    plt.close()


def update_and_logging(loss_t, loss_dict):
    txt = ''
    for key, value in loss_dict.items():
        loss_t[key] += value.item()
        txt += f'{key}: {value.item():.5f}  '
    return loss_t, txt


def logging(loss_t, denom):
    txt = ''
    for key, value in loss_t.items():
        loss_t[key] = value / denom
        txt += f'[{key}: {loss_t[key]:.5f}] '
    return txt
