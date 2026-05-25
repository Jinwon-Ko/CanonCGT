from engines.utils_train import get_loss_names, get_total_loss, update_and_logging, logging
from utils.util import to_np, logger, save_final_model
from utils.viz_utils import Visualizer


def train_one_epoch(cfg, epoch, model, train_loader, criterion, optimizer):
    print('Epoch %03d' % epoch)
    model.train()
    criterion.train()

    viz_tools = Visualizer()

    num = 0
    loss_t = get_loss_names(cfg)
    for i, (img, style_label) in enumerate(train_loader):
        img = img.flatten(0, 1).cuda()
        style_label = style_label.flatten(0, 1).cuda()

        # Forward model
        outputs = model(img)

        # Backpropagation
        batch = {'style_label': style_label}
        loss_dict = get_total_loss(cfg, criterion, batch, outputs)
        losses = sum(loss_dict[k] for k in loss_dict.keys())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Logging
        num += len(style_label)
        loss_t, log = update_and_logging(loss_t, loss_dict)
        loss_t['Total'] += losses.item()
        txt = 'Total: %.5f  %s' % (losses.item(), log)
        print('[Epoch %d][%d/%d][Losses %s]' % (epoch, i, len(train_loader), txt), end='\r')

    # # Visualize
    # if cfg.viz and (epoch % 5 == 0):
    #     viz_tools.viz_TSNE(cfg, style_feats, style_labels, mode='train', epoch=epoch)

    # logging
    log = logging(loss_t, num)
    logger("[Epoch %d Average Losses] %s\n" % (epoch, log), f'{cfg.save_dir}/losses_style_encoder.txt')
    print('\n[Epoch %d Average Losses] %s' % (epoch, log))
    save_final_model(cfg, model, epoch)

    return model, optimizer, loss_t


