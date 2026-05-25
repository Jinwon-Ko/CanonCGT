def load_dataset(cfg, mode=''):
    if 'style_encoder' in cfg.yaml:
        from dataloaders.Style_Library.style_encoder import Train_dataset, Test_dataset
        dataset = Train_dataset(cfg) if mode == 'train' else Test_dataset(cfg)

    if 'canonicalizer' in cfg.yaml:
        from dataloaders.Style_Library.canonicalizer import Train_dataset, Test_dataset
        dataset = Train_dataset(cfg) if mode == 'train' else Test_dataset(cfg)

    if 'styler' in cfg.yaml:
        from dataloaders.Style_Library.styler import Train_dataset, Test_dataset
        dataset = Train_dataset(cfg) if mode == 'train' else Test_dataset(cfg)

    if 'end_to_end_finetuning' in cfg.yaml:
        from dataloaders.Style_Library.end_to_end_finetuning import Train_dataset, Test_dataset
        dataset = Train_dataset(cfg) if mode == 'train' else Test_dataset(cfg)

    return dataset
