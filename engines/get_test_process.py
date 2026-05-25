def do_test_process(cfg, model):

    if 'style_encoder' in cfg.yaml:
        from engines.style_encoder.test_process import test_process

    if 'canonicalizer' in cfg.yaml:
        from engines.canonicalizer.test_process import test_process

    if 'styler' in cfg.yaml:
        from engines.styler.test_process import test_process

    if 'end_to_end_finetuning' in cfg.yaml:
        from engines.end_to_end_finetuning.test_process import test_process

    if 'SSL_training' in cfg.yaml:
        from engines.SSL_training.test_process import test_process

    test_process(cfg, model)
