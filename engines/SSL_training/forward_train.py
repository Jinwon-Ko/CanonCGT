from engines.utils_train import get_loss_names, get_total_loss, update_and_logging, logging
from utils.util import to_np, logger, save_final_model
from utils.viz_utils import Visualizer
from utils.calculate_metrics import Evaluator


def train_one_epoch_mixed(cfg, epoch, model, mixed_loader, criterion, optimizer):
    print('Epoch %03d' % epoch)
    model.train()
    criterion.train()

    model.Embedding_Net.eval()
    model.Embedding_Net.freeze_params_()

    viz_tools = Visualizer()
    eval_tools = Evaluator(cfg)

    num = 0
    psnr = 0
    loss_t = get_loss_names(cfg)
    for i, batch in enumerate(mixed_loader):

        # Load data
        batch = mixed_batch_reformatting(batch)

        # Forward model
        outputs = {}
        outputs.update(model.forward_Canonicalizer(batch['canonicalizer_input']))
        outputs.update(model.forward_Restyler(outputs['canonicalized'], batch['restyler_refer']))

        # Backpropagation
        loss_dict = get_total_loss(cfg, criterion, batch, outputs)
        losses = sum(loss_dict[k] for k in loss_dict.keys())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        psnr += eval_tools.measure_PSNR(outputs['restyled'], batch['restyler_GT'])

        # Visualize
        if cfg.viz and (i % 100 == 0):

            for b_idx in range(len(batch['restyler_GT'])):
                viz_contents = {'input': batch['canonicalizer_input'][b_idx],
                                'canonicalized': outputs['canonicalized'][b_idx],
                                'canonicalizer_GT': batch['canonicalizer_GT'][b_idx],
                                'restyler_refer': batch['restyler_refer'][b_idx],
                                'restyler_GT': batch['restyler_GT'][b_idx],
                                'restyled': outputs['restyled'][b_idx],
                                'canonical_refer': batch['canonical_refer'][b_idx]}

                viz_tools.viz_train(cfg, viz_contents, n_iter=num + b_idx)

        # Logging
        num += len(batch['restyler_GT'])
        loss_t, log = update_and_logging(loss_t, loss_dict)
        loss_t['Total'] += losses.item()
        txt = 'Total: %.5f  %s' % (losses.item(), log)
        print('[Epoch %d][%d/%d][Losses %s]' % (epoch, i, len(mixed_loader), txt), end='\r')

    # logging
    log = logging(loss_t, num)
    logger("[Epoch %d Average Losses] %s\n" % (epoch, log), f'{cfg.save_dir}/stage3_losses_style_transition.txt')
    print('\n[Epoch %d Average Losses] %s' % (epoch, log))
    save_final_model(cfg, model, epoch)

    PSNR = psnr / num
    logger("[Epoch %d]  Train ==> PSNR %5f\n" % (epoch, PSNR), f'{cfg.save_dir}/stage3_train_performances.txt')
    print('[Epoch %d]  Train ==> PSNR %5f' % (epoch, PSNR))
    return model, optimizer, loss_t


def mixed_batch_reformatting(batch):
    reformat_batch = {}
    reformat_batch['canonicalizer_input'] = batch['canonicalizer_input'].cuda()     # X
    reformat_batch['restyler_refer'] = batch['restyler_refer'].cuda()               # Y_gt (= Y_0)
    reformat_batch['restyler_GT'] = batch['restyler_GT'].cuda()                     # R
    reformat_batch['canonicalizer_GT'] = batch['canonicalizer_GT'].cuda()           # Z_gt
    reformat_batch['canonical_refer'] = batch['canonical_refer'].cuda()             # R_0
    reformat_batch['pair_type'] = batch['pair_type']

    return reformat_batch
