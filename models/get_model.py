def get_model(cfg):

    if 'style_encoder' in cfg.yaml:
        from models.networks.style_encoder import Net

    if 'canonicalizer' in cfg.yaml:
        from models.networks.canonicalizer import Net

    if 'styler' in cfg.yaml:
        from models.networks.styler import Net

    if 'end_to_end_finetuning' in cfg.yaml:
        from models.networks.end_to_end_finetuning import Net

    if 'SSL_training' in cfg.yaml:
        from models.networks.SSL_training import Net

    model = Net(cfg)
    return model
