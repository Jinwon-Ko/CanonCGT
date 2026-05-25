import os
import yaml
import torch
import argparse

from torch.utils.data import DataLoader

from dataloaders.Style_Library.SSL_training import Supervised_dataset_eval, Unsupervised_dataset_eval
from models.networks.SSL_training import CanonCGT_SSL
from utils.calculate_metrics import Evaluator


def override_config_with_yaml(args, yaml_path):
    if not os.path.exists(yaml_path):
        print(f"[Warning] yaml file {yaml_path} not found. Using default config.")
        return args

    with open(yaml_path, "r") as f:
        override_args = yaml.safe_load(f)

    args_dict = vars(args)
    for k, v in override_args.items():
        args_dict[k] = v

    return argparse.Namespace(**args_dict)


def evaluate_SSL(cfg, model, loader):
    eval_tools = Evaluator(cfg)
    eval_tools.cuda().eval()

    with torch.no_grad():

        metrics_by_source = {}

        for i, batch in enumerate(loader):
            torch.cuda.empty_cache()

            print('Processing [%04d/%04d]...' % (i, len(loader)), end='\r')

            # Load data
            dataset_name = batch['dataset_name']
            inp = batch['inp'].cuda()
            ref = batch['ref'].cuda()
            gt = batch['gt'].cuda()

            # Forward model
            outputs = model(inp, ref)
            pred = outputs['restyled']

            # Evaluation Fidelity (GT <-> Output)
            metrics = {'PSNR': eval_tools.measure_PSNR(pred, gt),
                       'SSIM': eval_tools.measure_SSIM(pred, gt),
                       'DeltaEab': eval_tools.measure_DeltaEab(pred, gt),
                       'LPIPS': eval_tools.measure_LPIPS(pred, gt)}
                       #'SSIM_ED': eval_tools.measure_SSIM_ED(pred, gt),
                       #'H_Corr': eval_tools.measure_HCorr(pred, gt),
                       #'H_Chi': eval_tools.measure_HChi(pred, gt)

            src = dataset_name[0]
            if src not in metrics_by_source:
                metrics_by_source[src] = {k: [] for k in metrics.keys()}
            for k, v in metrics.items():
                metrics_by_source[src][k].append(v)

    # mean Metric
    print("\n\n==== Per-dataset results ====")
    for src, m in metrics_by_source.items():
        means = {k: sum(v) / len(v) for k, v in m.items()}
        print(f"\n[{src.upper()}] ({len(list(m.values())[0])} samples)")
        for k, v in means.items():
            print(f"  {k}: {v:.5f}")

    print("\n==== Overall average ====")
    all_metrics = {k: [] for k in list(metrics_by_source.values())[0].keys()}
    for src, m in metrics_by_source.items():
        for k in all_metrics.keys():
            all_metrics[k].extend(m[k])
    overall_mean = {k: sum(v) / len(v) for k, v in all_metrics.items()}
    for k, v in overall_mean.items():
        print(f"  {k}: {v:.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--yaml_path", type=str, default="./configs/Stage3_SSL_training_Flickr2K_PPR10K_LSDIR.yaml")
    parser.add_argument("--pretrained_path", type=str, default="./pretrained/SSL_updated_251111.pth")
    parser.add_argument("--dataset_root", type=str, default="/media/jwko/b0376b00-2c8f-472a-a29e-fd95b8a02058/Datasets") # Set Your Dataset Root

    cfg = parser.parse_args()
    cfg = override_config_with_yaml(cfg, cfg.yaml_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

    model = CanonCGT_SSL(cfg)
    checkpoint = torch.load(cfg.pretrained_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    checkpoint = None

    model.cuda()
    model.eval()

    # print("[INFO] Start Evaluation SSL trained model on Supervised dataset")
    # Supervised_testset = Supervised_dataset_eval(dataset_root=cfg.dataset_root)
    # Supervised_loader = DataLoader(Supervised_testset, batch_size=20, num_workers=cfg.dataset['num_workers'], shuffle=False)
    # evaluate_SSL(cfg, model, Supervised_loader)

    print("[INFO] Start Evaluation SSL trained model on Unsupervised dataset")
    g = torch.Generator()
    g.manual_seed(42)
    Unsupervised_testset = Unsupervised_dataset_eval(dataset_root=cfg.dataset_root)
    Unsupervised_loader = DataLoader(Unsupervised_testset, batch_size=1, num_workers=cfg.dataset['num_workers'], shuffle=False, generator=g)
    evaluate_SSL(cfg, model, Unsupervised_loader)

if __name__ == "__main__":
    main()
