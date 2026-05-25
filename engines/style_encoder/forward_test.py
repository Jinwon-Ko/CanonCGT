import os
import math
import time
import torch
import numpy as np

from utils.util import to_np, save_best_model, logger, save_pickle
from utils.viz_utils import Visualizer
from utils.calculate_metrics import Evaluator_SupCon


def evaluation(cfg, epoch, model, test_loader, best, now):
    if (epoch + 1) < cfg.testing['start_eval_epoch']:
        return best, now
    
    if (epoch + 1) % cfg.testing['eval_epoch'] != 0:
        return best, now

    model.eval()

    eval_tools = Evaluator_SupCon(cfg)
    viz_tools = Visualizer()

    style_feats = []
    style_labels = []
    for i, (img, style_label) in enumerate(test_loader):
        print('[Epoch: %d][%d/%d]' % (epoch, i, len(test_loader)), end='\r')

        # load data
        img = img.cuda()
        style_label = style_label.cuda()

        with torch.no_grad():
            outputs = model(img)

        style_feats.append(outputs['style_vector'])
        style_labels.append(style_label)

    style_feats = torch.cat(style_feats, dim=0)
    style_labels = torch.cat(style_labels, dim=0)

    acc = eval_tools.measure_knn_accuracy(style_feats, style_labels)

    if cfg.viz:
        viz_tools.viz_TSNE(cfg, style_feats, style_labels, mode='test', epoch=epoch)

    now = {'acc': acc}
    print('%s Test ==> ACC %5f' % (cfg.dataset_name, acc))
    best = save_best_model(cfg, model, epoch, now, best, metric='acc')
    return best, now


def analysis(cfg, model, test_loader):
    from sklearn.metrics import confusion_matrix

    model.eval()

    eval_tools = Evaluator_SupCon(cfg)
    viz_tools = Visualizer()

    style_dicts = test_loader.dataset.style_dicts

    style_feats = []
    style_labels = []
    with torch.no_grad():
        torch.cuda.empty_cache()

        for i, (img, style_label) in enumerate(test_loader):
            print('Processing [%04d/%04d]...' % (i, len(test_loader)), end='\r')

            # load data
            img = img.cuda()
            style_label = style_label.cuda()

            outputs = model(img)

            style_feats.append(outputs['style_vector'])
            style_labels.append(style_label)

    style_feats = torch.cat(style_feats, dim=0)
    style_labels = torch.cat(style_labels, dim=0)

    # Total accuracy
    acc, preds = eval_tools.measure_knn_accuracy(style_feats, style_labels, return_pred=True)
    print('%s ==> [Prediction] Total [Acc %.5f]' % (cfg.dataset_name, acc))
    print('# of trainable parameters : %.3f K' % (sum(p.numel() for p in model.parameters() if p.requires_grad) * 0.001))

    if cfg.viz:
        viz_tools.viz_TSNE(cfg, style_feats, style_labels, mode='analysis', epoch=0)

    labels_np = to_np(style_labels)

    # Confusion matrix
    style_ids = sorted([int(style_idx) for style_idx in style_dicts.keys()])
    cm = confusion_matrix(labels_np, preds, labels=style_ids)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    header = '\t'.join([f'[{style_idx}]' for style_idx in style_dicts.keys()])
    logger(f'\t{header}\n', f'{cfg.save_dir}/Confusion_Matrix.txt')
    for row, gt_style in enumerate(style_dicts.keys()):
        row_list = []
        for col, pred_style in enumerate(style_dicts.keys()):
            row_list.append(f'{cm_normalized[row, col] * 100.:.2f}')
        row = '\t'.join(row_list)

        logger(f'[{gt_style}]:\t{row}\n', f'{cfg.save_dir}/Confusion_Matrix.txt')

    centroids = {}
    feats_np = to_np(style_feats)
    unique_labels = sorted(set(labels_np.tolist()))
    for cls in unique_labels:
        cls_feats = feats_np[labels_np == cls]
        cls_centroid = cls_feats.mean(axis=0)
        centroids[int(cls)] = cls_centroid

    save_pickle(dir_name=cfg.output_dir, file_name='style_centroids.pickle', data=centroids)



def save_style_centroids(cfg):
    from models.networks.style_encoder import Net
    from dataloaders.Style_Library.style_encoder import Test_dataset
    from torch.utils.data import DataLoader

    print("[Info] Style Centroids Update!!!")
    test_dataset = Test_dataset(cfg)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=cfg.dataset['num_workers'], shuffle=False)

    model = Net(cfg)
    ckpt_path = os.path.join(cfg.output_dir, 'Stage1_style_encoder/weights/ckpt/checkpoint_best.pth')
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    checkpoint = None

    model.cuda()
    model.eval()

    style_feats = []
    style_labels = []
    with torch.no_grad():
        torch.cuda.empty_cache()

        for i, (img, style_label) in enumerate(test_loader):
            print('Processing [%04d/%04d]...' % (i, len(test_loader)), end='\r')

            # load data
            img = img.cuda()
            style_label = style_label.cuda()

            outputs = model(img)

            style_feats.append(outputs['style_vector'])
            style_labels.append(style_label)

    style_feats = torch.cat(style_feats, dim=0)
    style_labels = torch.cat(style_labels, dim=0)

    centroids = {}
    feats_np = to_np(style_feats)
    labels_np = to_np(style_labels)
    unique_labels = sorted(set(labels_np.tolist()))
    for cls in unique_labels:
        cls_feats = feats_np[labels_np == cls]
        cls_centroid = cls_feats.mean(axis=0)
        centroids[int(cls)] = cls_centroid

    dir_name, file_name = os.path.split(cfg.checkpoint['centroids'])
    save_pickle(dir_name=dir_name, file_name=file_name, data=centroids)

