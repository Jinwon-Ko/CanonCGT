import os
import yaml
import argparse


def write_log(log_file, out_str):
    log_file.write(out_str + '\n')
    log_file.flush()
    # print(out_str)


class Config:
    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('--gpu', type=str, default='0')
        parser.add_argument('--yaml', type=str, default='Stage3_SSL_training_Flickr2K_PPR10K_LSDIR', help='YAML filename in configs/')
        parser.add_argument('--host', type=str, default='desk', choices=['desk', 'server'])
        parser.add_argument('--run_mode', type=str, default='train', choices=['train', 'test', 'eval'])
        parser.add_argument('--load', action='store_true', help='Load pretrained models')
        parser.add_argument('--viz', action='store_true', help='Visualize results')
        parser.add_argument('--eval', action='store_true', help='Load pretrained models')
        parser.add_argument('--desc', type=str, help='Description for experimental settings')
        args = parser.parse_args()

        for k, v in vars(args).items():
            setattr(self, k, v)

        self.root = os.path.abspath(os.path.join(os.getcwd(), '..'))

        self.dataset_name = 'Style_transition'
        self.override_config_with_yaml(f'configs/{self.yaml}.yaml')
        self.settings_for_path()
        if self.run_mode == 'train':
            self.log_configs()

    def settings_for_path(self):
        if self.host == 'desk':
            # self.dataset_root = os.path.abspath(os.path.join('/home/jwko/Datasets'))
            self.dataset_root = os.path.abspath(os.path.join('/media/jwko/b0376b00-2c8f-472a-a29e-fd95b8a02058/Datasets'))
        elif self.host == 'server':
            self.dataset_root = os.path.abspath(os.path.join('/hdd1/jwko/Datasets'))

        self.code_name = os.getcwd().split('/')[-1]
        self.output_name = self.code_name.replace('_code', '_output')
        self.proj_dir = os.path.join(self.root, f'{self.code_name}')
        self.output_dir = os.path.join(self.root, f'{self.output_name}')

        self.viz_dir = os.path.join(self.output_dir, f'{self.yaml}/display')
        self.save_dir = os.path.join(self.output_dir, f'{self.yaml}/weights')
        os.makedirs(self.viz_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        self.checkpoint = {}
        if 'style_encoder' not in self.yaml:
            self.checkpoint['centroids'] = os.path.join(self.output_dir, 'style_centroids.pickle')

        checkpoint = 'weights/ckpt/checkpoint_best.pth'
        if self.load:
            self.checkpoint['model'] = os.path.join(self.output_dir, self.yaml, checkpoint)

        else:
            if 'end_to_end' in self.yaml:
                self.checkpoint['Embedding_network'] = os.path.join(self.output_dir, 'Stage1_style_encoder', checkpoint)
                self.checkpoint['Canonicalizer'] = os.path.join(self.output_dir, 'Stage1_canonicalizer', checkpoint)
                self.checkpoint['Restyler'] = os.path.join(self.output_dir, 'Stage1_styler', checkpoint)

            if 'SSL_training' in self.yaml:
                self.checkpoint['model'] = os.path.join(self.output_dir, 'Stage2_end_to_end_finetuning', checkpoint)

    def override_config_with_yaml(self, yaml_path):
        with open(yaml_path, 'r') as f:
            override_cfg = yaml.safe_load(f)
        self._recursive_update(self.__dict__, override_cfg)

    def _recursive_update(self, base, overrides):
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                self._recursive_update(base[k], v)
            else:
                base[k] = v

    def log_configs(self, log_file='log.txt'):
        if os.path.exists(f'{self.save_dir}/{log_file}'):
            log_file = open(f'{self.save_dir}/{log_file}', 'a')
        else:
            log_file = open(f'{self.save_dir}/{log_file}', 'w')

        write_log(log_file, '------------ Options -------------')
        for k in vars(self):
            write_log(log_file, f'{str(k)}: {str(vars(self)[k])}')
        write_log(log_file, '-------------- End ----------------')

        log_file.close()
        return
