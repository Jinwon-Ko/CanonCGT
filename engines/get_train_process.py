
def do_train_process(cfg, model, criterion, optimizer, lr_scheduler):

    if 'style_encoder' in cfg.yaml:
        from engines.style_encoder.train_process import train_process

    if 'canonicalizer' in cfg.yaml:
        from engines.canonicalizer.train_process import train_process

    if 'styler' in cfg.yaml:
        from engines.styler.train_process import train_process

    if 'end_to_end_finetuning' in cfg.yaml:
        from engines.end_to_end_finetuning.train_process import train_process

    if 'SSL_training' in cfg.yaml:
        from engines.SSL_training.train_process import train_process

    train_process(cfg, model, criterion, optimizer, lr_scheduler)
