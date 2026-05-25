import os
import shutil
import random
import pickle
from datetime import datetime
from collections import OrderedDict

import numpy as np
import torch


def logger(text, LOGGER_FILE):  # write log
    with open(LOGGER_FILE, 'a') as f:
        f.write(text),
        f.close()

# convertor
def to_tensor(data):
    try:
        return torch.from_numpy(data).cuda()
    except:
        return torch.from_numpy(data)

def to_np(data):
    try:
        return data.cpu().numpy()
    except:
        return data.detach().cpu().numpy()


# pickle
def save_pickle(dir_name, file_name, data):
    os.makedirs(dir_name, exist_ok=True)
    with open(os.path.join(dir_name, file_name), 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


# Checkpoints
def save_final_model(args, net, epoch):
    save_dir = os.path.join(args.save_dir, 'ckpt')
    os.makedirs(save_dir, exist_ok=True)

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': net.state_dict(),
    }, os.path.join(save_dir + '/' + f'checkpoint_final.pth'))


def save_best_model(args, net, epoch, now, best, metric='PSNR'):
    greater_is_best = ['PSNR', 'SSIM', 'acc']
    save_dir = os.path.join(args.save_dir, 'ckpt')
    os.makedirs(save_dir, exist_ok=True)

    epochs = args.training['epochs']
    if metric in greater_is_best:
        if now[metric] > best[metric]:
            best[metric] = now[metric]

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
            }, os.path.join(save_dir + '/' + f'checkpoint_best.pth'))
            print('Saved best_model to ' + save_dir + '/' + f'checkpoint_best.pth')
            print(f'Epoch [{epoch:03d}/{epochs:03d}] Best model performances : ' + f'[{metric} {now[metric]:.5f}]')
            logger(f"Average Metrics : [Epoch: {epoch:03d}] [{metric}: {now[metric]:.5f} ]\n", f'{args.save_dir}/results.txt')
        return best
    else:
        if now[metric] < best[metric]:
            best[metric] = now[metric]

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
            }, os.path.join(save_dir + '/' + f'checkpoint_best.pth'))
            print('Saved best_model to ' + save_dir + '/' + f'checkpoint_best.pth')
            print(f'Epoch [{epoch:03d}/{epochs:03d}] Best model performances : ' + f'[{metric} {now[metric]:.5f}]')
            logger(f"Average Metrics : [Epoch: {epoch:03d}] [{metric}: {now[metric]:.5f} ]\n",
                   f'{args.save_dir}/results.txt')
        return best


def copy_code(args):
    if os.path.exists(os.path.join(args.save_dir, 'code')):
        shutil.rmtree(os.path.join(args.save_dir, 'code'))
    os.makedirs(os.path.join(args.save_dir, 'train/configs'), exist_ok=True)
    shutil.copytree(os.path.join(args.proj_dir),
                    os.path.join(args.save_dir, 'code'))
