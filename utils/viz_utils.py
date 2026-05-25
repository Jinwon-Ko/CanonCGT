import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from utils.util import to_np, to_tensor


############################################
class Visualizer(object):
    def __init__(self):
        self.show = {}

    def update_image(self, img, name='img'):
        try:
            img = to_np(img.permute(1, 2, 0))
            img = np.clip(img, a_min=0.0, a_max=1.0)
            if 'error' in name:
                img = img ** 0.5
            img = np.round(img * 255)
            img = np.uint8(img)[:, :, [2, 1, 0]]
            if 'error' in name:
                img  = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        except:
            pass

        self.show[name] = img

    def saveimg_one(self, dir_name, file_name, show_name):
        os.makedirs(dir_name, exist_ok=True)
        if ('.png' in file_name) or ('.jpg' in file_name):
            save_name = file_name
        else:
            name, ext = os.path.splitext(file_name)
            save_name = name + '.png'
        cv2.imwrite(os.path.join(dir_name, save_name), self.show[show_name])

    def saveimg_list(self, dir_name, file_name, show_list):
        # boundary line

        if self.show[show_list[0]].shape[-1] == 3:
            line = np.zeros((self.show[show_list[0]].shape[0], 3, 3), dtype=np.uint8)
            line[:, :, :] = 255
        else:
            line = np.zeros((self.show[show_list[0]].shape[0], 3), dtype=np.uint8)
            line[:, :] = 255
        disp = line

        for i in range(len(show_list)):
            if show_list[i] not in self.show.keys():
                continue
            disp = np.concatenate((disp, self.show[show_list[i]], line), axis=1)

        os.makedirs(dir_name, exist_ok=True)
        cv2.imwrite(os.path.join(dir_name, file_name), disp)

    def saveimg_dict(self, dir_name, file_name, show_dict):
        for idx, key in enumerate(show_dict.keys()):
            show_list = show_dict[key]

            # boundary line
            if self.show[show_list[0]].shape[-1] == 3:
                line = np.zeros((self.show[show_list[0]].shape[0], 3, 3), dtype=np.uint8)
                line[:, :, :] = 255
            else:
                line = np.zeros((self.show[show_list[0]].shape[0], 3), dtype=np.uint8)
                line[:, :] = 255

            # stack images by column direction
            col_disp = line
            for i in range(len(show_list)):
                if show_list[i] not in self.show.keys():
                    continue
                col_disp = np.concatenate((col_disp, self.show[show_list[i]], line), axis=1)

            # stack images by row direction
            self.row_line = np.ones((3, col_disp.shape[1], 3), dtype=np.uint8) * 255
            if idx == 0:
                disp = self.row_line
            disp = np.concatenate((disp, col_disp, self.row_line), axis=0)

        # save image
        os.makedirs(dir_name, exist_ok=True)
        cv2.imwrite(os.path.join(dir_name, file_name), disp)

    def viz_train(self, cfg, viz_contents, n_iter):

        if 'SSL_training' in cfg.yaml:
            # Draw input & output & GT
            self.update_image(img=viz_contents['input'], name='input')
            self.update_image(img=viz_contents['canonicalized'], name='canonicalized')
            self.update_image(img=viz_contents['restyler_refer'], name='restyler_refer')
            self.update_image(img=viz_contents['restyled'], name='restyled')
            self.update_image(img=viz_contents['restyler_GT'], name='restyler_GT')
            self.update_image(img=abs(viz_contents['restyler_GT'] - viz_contents['restyled']), name='restyled_error')

            show_dict = {'row1': ['input',         'restyler_refer', 'restyler_GT'],
                         'row2': ['canonicalized', 'restyled',       'restyled_error']}
            # [X, R, Z_gt]
            # [Y, Z, error]
            dir_name = os.path.join(cfg.viz_dir, f'train/results')
            self.saveimg_dict(dir_name=os.path.join(dir_name), file_name=f'iter_{n_iter:04d}.jpg', show_dict=show_dict)

        elif 'end_to_end' in cfg.yaml:
            # Draw input & output & GT
            self.update_image(img=viz_contents['input'], name='input')
            self.update_image(img=viz_contents['canonicalized'], name='canonicalized')
            self.update_image(img=viz_contents['canonicalizer_GT'], name='canonicalizer_GT')
            self.update_image(img=abs(viz_contents['canonicalizer_GT'] - viz_contents['canonicalized']), name='canonicalized_error')

            self.update_image(img=viz_contents['restyler_refer'], name='restyler_refer')
            self.update_image(img=viz_contents['restyled'], name='restyled')
            self.update_image(img=viz_contents['restyler_GT'], name='restyler_GT')
            self.update_image(img=abs(viz_contents['restyler_GT'] - viz_contents['restyled']), name='restyled_error')
            self.update_image(img=viz_contents['canonical_refer'], name='canonical_refer')

            show_dict = {'row1': ['input',           'canonicalized',        'restyled'],
                         'row2': ['canonical_refer', 'canonicalized_error',  'restyled_error'],
                         'row3': ['restyler_refer',  'canonicalizer_GT',     'restyler_GT']}

            dir_name = os.path.join(cfg.viz_dir, f'train/results')
            self.saveimg_dict(dir_name=os.path.join(dir_name), file_name=f'iter_{n_iter:04d}.jpg', show_dict=show_dict)

    def viz_test(self, cfg, viz_contents, img_name):

        if 'SSL_training' in cfg.yaml:
            # Draw input & output & GT
            self.update_image(img=viz_contents['input'], name='input')
            self.update_image(img=viz_contents['canonicalized'], name='canonicalized')
            self.update_image(img=viz_contents['restyler_refer'], name='restyler_refer')
            self.update_image(img=viz_contents['restyled'], name='restyled')
            self.update_image(img=viz_contents['restyler_GT'], name='restyler_GT')
            self.update_image(img=abs(viz_contents['restyler_GT'] - viz_contents['restyled']), name='restyled_error')

            show_dict = {'row1': ['input',         'restyler_refer', 'restyler_GT'],
                         'row2': ['canonicalized', 'restyled',       'restyled_error']}
            # [X, R, Z_gt]
            # [Y, Z, error]
            name, ext = os.path.splitext(img_name)
            dir_name = os.path.join(cfg.viz_dir, f'test/results')
            self.saveimg_dict(dir_name=os.path.join(dir_name), file_name=name + '.jpg', show_dict=show_dict)

        elif 'end_to_end' in cfg.yaml:
            # Draw input & output & GT
            self.update_image(img=viz_contents['input'], name='input')
            self.update_image(img=viz_contents['canonicalized'], name='canonicalized')
            self.update_image(img=viz_contents['canonicalizer_GT'], name='canonicalizer_GT')
            self.update_image(img=abs(viz_contents['canonicalizer_GT'] - viz_contents['canonicalized']), name='canonicalized_error')

            self.update_image(img=viz_contents['restyler_refer'], name='restyler_refer')
            self.update_image(img=viz_contents['restyled'], name='restyled')
            self.update_image(img=viz_contents['restyler_GT'], name='restyler_GT')
            self.update_image(img=abs(viz_contents['restyler_GT'] - viz_contents['restyled']), name='restyled_error')
            self.update_image(img=viz_contents['canonical_refer'], name='canonical_refer')

            show_dict = {'row1': ['input',              'canonicalized',        'restyled'],
                         'row2': ['canonical_refer',    'canonicalized_error',  'restyled_error'],
                         'row3': ['restyler_refer',     'canonicalizer_GT',     'restyler_GT']}
            name, ext = os.path.splitext(img_name)
            dir_name = os.path.join(cfg.viz_dir, f'test/results')
            self.saveimg_dict(dir_name=os.path.join(dir_name), file_name=name + '.jpg', show_dict=show_dict)

    def viz_analysis(self, cfg, viz_contents, tag, img_name):

        if 'SSL_training' in cfg.yaml:
            # Draw input & output & GT
            self.update_image(img=viz_contents['input'], name='input')
            self.update_image(img=viz_contents['canonicalized'], name='canonicalized')
            self.update_image(img=viz_contents['restyler_refer'], name='restyler_refer')
            self.update_image(img=viz_contents['restyled'], name='restyled')
            self.update_image(img=viz_contents['restyler_GT'], name='restyler_GT')
            self.update_image(img=abs(viz_contents['restyler_GT'] - viz_contents['restyled']), name='restyled_error')

            show_dict = {'row1': ['input',         'restyler_refer', 'restyler_GT'],
                         'row2': ['canonicalized', 'restyled',       'restyled_error']}
            # [X, R, Z_gt]
            # [Y, Z, error]
            name, ext = os.path.splitext(img_name)
            dir_name = os.path.join(cfg.viz_dir, f'analysis/results')
            self.saveimg_dict(dir_name=os.path.join(dir_name, tag), file_name=name + '.jpg', show_dict=show_dict)

        elif 'end_to_end' in cfg.yaml:
            # Draw input & output & GT
            self.update_image(img=viz_contents['input'], name='input')
            self.update_image(img=viz_contents['canonicalized'], name='canonicalized')
            self.update_image(img=viz_contents['canonicalizer_GT'], name='canonicalizer_GT')
            self.update_image(img=abs(viz_contents['canonicalizer_GT'] - viz_contents['canonicalized']), name='canonicalized_error')

            self.update_image(img=viz_contents['restyler_refer'], name='restyler_refer')
            self.update_image(img=viz_contents['restyled'], name='restyled')
            self.update_image(img=viz_contents['restyler_GT'], name='restyler_GT')
            self.update_image(img=abs(viz_contents['restyler_GT'] - viz_contents['restyled']), name='restyled_error')
            self.update_image(img=viz_contents['canonical_refer'], name='canonical_refer')

            show_dict = {'row1': ['input',              'canonicalized',        'restyled'],
                         'row2': ['canonical_refer',    'canonicalized_error',  'restyled_error'],
                         'row3': ['restyler_refer',     'canonicalizer_GT',     'restyler_GT']}

            name, ext = os.path.splitext(img_name)
            dir_name = os.path.join(cfg.viz_dir, f'analysis/results')
            self.saveimg_dict(dir_name=os.path.join(dir_name, tag), file_name=name + '.jpg', show_dict=show_dict)

    def viz_demo(self, cfg, viz_contents, file_name, mode='one'):
        ######################################################################################
        # Draw input & output & GT
        self.update_image(img=viz_contents['inp'], name='inp')
        self.update_image(img=viz_contents['ref'], name='ref')
        self.update_image(img=viz_contents['out'], name='out')

        if mode == 'one':
            self.saveimg_one(dir_name=os.path.join(cfg.viz_dir, 'demo/inputs'), file_name=file_name, show_name='inp')
            self.saveimg_one(dir_name=os.path.join(cfg.viz_dir, 'demo/refers'), file_name=file_name, show_name='ref')
            self.saveimg_one(dir_name=os.path.join(cfg.viz_dir, 'demo/proposed'), file_name=file_name, show_name='out')

        elif mode == 'dict':
            show_dict = {'row1': ['inp', 'ref', 'gt']}
            self.saveimg_dict(dir_name=os.path.join(cfg.viz_dir, 'demo/pairs'), file_name=file_name, show_dict=show_dict)

    def viz_comparison_per_style_imgs(self, cfg, img_name):
        ######################################################################################
        # Draw input & output & GT
        show_dict = {'row1': ['gt',    'input_01', 'input_02', 'input_03', 'input_04', 'input_05'],
                     'row2': ['empty', 'input_06', 'input_07', 'input_08', 'input_09', 'input_10'],
                     'row3': ['empty', 'input_11', 'input_12', 'input_13', 'input_14', 'input_15'],
                     'row4': ['empty', 'input_16', 'input_17', 'input_18', 'input_19', 'input_20']}

        name, ext = os.path.splitext(img_name)
        dir_name = os.path.join(cfg.viz_dir, f'comparison_per_expert/inputs')
        self.saveimg_dict(dir_name=os.path.join(dir_name), file_name=name + '.jpg', show_dict=show_dict)

    def viz_comparison_per_style_preds(self, cfg, img_name):
        ######################################################################################
        # Draw input & output & GT
        show_dict = {'row1': ['gt',    'output_01', 'output_02', 'output_03', 'output_04', 'output_05'],
                     'row2': ['empty', 'output_06', 'output_07', 'output_08', 'output_09', 'output_10'],
                     'row3': ['empty', 'output_11', 'output_12', 'output_13', 'output_14', 'output_15'],
                     'row4': ['empty', 'output_16', 'output_17', 'output_18', 'output_19', 'output_20']}

        name, ext = os.path.splitext(img_name)
        dir_name = os.path.join(cfg.viz_dir, f'comparison_per_expert/results')
        self.saveimg_dict(dir_name=os.path.join(dir_name), file_name=name + '.jpg', show_dict=show_dict)

    def viz_TSNE(self, cfg, feats, labels, mode, epoch):
        from sklearn.manifold import TSNE
        from matplotlib.colors import ListedColormap

        feats = to_np(feats)
        labels = to_np(labels)

        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, init='random', random_state=42)
        feats_2d = tsne.fit_transform(feats)

        num_classes = len(np.unique(labels))
        colors = plt.get_cmap('tab20').colors[:num_classes]
        cmap = ListedColormap(colors)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(feats_2d[:, 0], feats_2d[:, 1], c=labels, cmap=cmap, s=8, alpha=0.8)
        plt.colorbar(scatter, ticks=range(num_classes))
        plt.title('t-SNE')
        plt.xlabel('dim 1')
        plt.ylabel('dim 2')
        plt.grid(True)
        plt.tight_layout()

        dir_name = os.path.join(cfg.viz_dir, f'{mode}/tSNE')
        os.makedirs(dir_name, exist_ok=True)
        file_name = f'epoch_{epoch:04d}_all.jpg'
        plt.savefig(os.path.join(dir_name, file_name), dpi=300)
        plt.close()

        original_style_idx = (labels == 0)
        feats_2d_original = feats_2d[original_style_idx]
        label0_color = colors[0]

        plt.figure(figsize=(10, 8))
        plt.scatter(feats_2d[:, 0], feats_2d[:, 1], c='lightgray', s=8, alpha=0.3, label='Others')
        plt.scatter(feats_2d_original[:, 0], feats_2d_original[:, 1], c=[label0_color], s=10, alpha=0.9, label='Original')
        plt.title('t-SNE (Original Highlighted)')
        plt.xlabel('dim 1')
        plt.ylabel('dim 2')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        file_name = f'epoch_{epoch:04d}_label0.jpg'
        plt.savefig(os.path.join(dir_name, file_name), dpi=300)
        plt.close()

