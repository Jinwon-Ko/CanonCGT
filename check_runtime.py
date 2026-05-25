import os
import yaml
import torch
import argparse

from models.networks.SSL_training import CanonCGT_SSL
from engines.check_complexities import check_runtime, check_complexities


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--yaml_path", type=str, default="./configs/Stage3_SSL_training_Flickr2K_PPR10K_LSDIR.yaml")

    cfg = parser.parse_args()
    cfg = override_config_with_yaml(cfg, cfg.yaml_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

    model = CanonCGT_SSL(cfg)

    model.cuda()
    model.eval()

    print("[INFO] Start Runtime Evaluation")
    check_runtime(model)            # GPU runtime & Model size
    check_complexities(model)       # GPU memory allocation