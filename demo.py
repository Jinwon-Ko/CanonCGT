import os
import yaml
import torch
import argparse

from PIL import Image
import torchvision.transforms as transforms
from models.networks.SSL_training import CanonCGT_SSL
from utils.viz_utils import Visualizer


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


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--yaml_path", type=str, default="./configs/Stage3_SSL_training_Flickr2K_PPR10K_LSDIR.yaml")
parser.add_argument("--pretrained_path", type=str, default="./pretrained/SSL_updated_251111.pth")
parser.add_argument("--inp_path", type=str, default="./samples/inp/00.png")     # Set your inp path
parser.add_argument("--ref_path", type=str, default="./samples/ref/00.png")     # Set your ref path
parser.add_argument("--out_path", type=str, default="./samples/out/00.png")     # Set your out path

cfg = parser.parse_args()
cfg = override_config_with_yaml(cfg, cfg.yaml_path)
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu

model = CanonCGT_SSL(cfg)
checkpoint = torch.load(cfg.pretrained_path)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
checkpoint = None

model.cuda()
model.eval()

viz_tools = Visualizer()
to_tensor = transforms.ToTensor()

with torch.no_grad():
    torch.cuda.empty_cache()

    inp = to_tensor(Image.open(cfg.inp_path).convert("RGB")).cuda()
    ref = to_tensor(Image.open(cfg.ref_path).convert("RGB")).cuda()

    # Load data
    model_input = inp.unsqueeze(0).cuda()
    model_refer = ref.unsqueeze(0).cuda()

    # Forward model
    outputs = model(model_input, model_refer)
    model_output = outputs['restyled']

    viz_contents = {'inp': model_input[0],
                    'ref': model_refer[0],
                    'out': model_output[0]}

    out_dir, out_fname = os.path.split(cfg.out_path)
    viz_tools.update_image(img=viz_contents['out'], name='out')
    viz_tools.saveimg_one(dir_name=out_dir, file_name=out_fname, show_name='out')
