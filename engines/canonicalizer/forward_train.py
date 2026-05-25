from engines.utils_train import get_loss_names, get_total_loss, update_and_logging, logging
from utils.util import to_np, logger, save_final_model
from utils.viz_utils import Visualizer
from utils.calculate_metrics import Evaluator



def train_one_epoch(cfg, epoch, model, train_loader, criterion, optimizer):
    print('Epoch %03d' % epoch)
    model.train()
    criterion.train()

    viz_tools = Visualizer()
    eval_tools = Evaluator(cfg)

    num = 0
    psnr = 0
    loss_t = get_loss_names(cfg)
    for i, batch in enumerate(train_loader):

        # Load data
        img = batch['img'].cuda()
        gt = batch['gt'].cuda()
        style_idx = batch['style_idx']

        # Forward model
        outputs = model(img, style_idx)

        # Backpropagation
        batch = {'gt': gt}
        loss_dict = get_total_loss(cfg, criterion, batch, outputs)
        losses = sum(loss_dict[k] for k in loss_dict.keys())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        psnr += eval_tools.measure_PSNR(outputs['canonicalized'], gt)

        # Visualize
        if cfg.viz and (i % 100 == 0):
            viz_contents = {'input': img[0],
                            'canonicalizer_GT': gt[0],
                            'canonicalized': outputs['canonicalized'][0]}
            viz_tools.viz_train(cfg, viz_contents, n_iter=num)

        # Logging
        num += len(img)
        loss_t, log = update_and_logging(loss_t, loss_dict)
        loss_t['Total'] += losses.item()
        txt = 'Total: %.5f  %s' % (losses.item(), log)
        print('[Epoch %d][%d/%d][Losses %s]' % (epoch, i, len(train_loader), txt), end='\r')

    # logging
    log = logging(loss_t, num)
    logger("[Epoch %d Average Losses] %s\n" % (epoch, log), f'{cfg.save_dir}/losses_canonicalizer.txt')
    print('\n[Epoch %d Average Losses] %s' % (epoch, log))
    save_final_model(cfg, model, epoch)

    PSNR = psnr / num
    logger("[Epoch %d]  Train ==> PSNR %5f\n" % (epoch, PSNR), f'{cfg.save_dir}/train_performances_canonicalizer.txt')
    print('[Epoch %d]  Train ==> PSNR %5f' % (epoch, PSNR))
    return model, optimizer, loss_t


