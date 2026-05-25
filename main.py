import os
from configs.config import Config
from models.get_model import get_model
from engines.loss import get_loss_function
from engines.optimizer import get_optimizer, get_scheduler
from engines.get_train_process import do_train_process
from engines.get_test_process import do_test_process
from engines.get_eval_process import do_eval_process

from utils.util import copy_code


if __name__ == '__main__':
    args = Config()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = get_model(args)
    criterion = get_loss_function(args)
    if 'train' in args.run_mode:
        copy_code(args)
        optimizer = get_optimizer(args, model)
        lr_scheduler = get_scheduler(args, optimizer)
        do_train_process(args, model, criterion, optimizer, lr_scheduler)

    elif 'test' in args.run_mode:           # For test
        do_test_process(args, model)

    elif 'eval' in args.run_mode:
        do_eval_process(args, model)
