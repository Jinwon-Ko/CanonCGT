import math
import torch
import torch.optim as optim


def get_optimizer(cfg, net):
    scheduler = cfg.training['scheduler']
    base_lr = cfg.training['base_lr']
    lr_min = cfg.training['lr_min']
    lr = lr_min if 'Cosine' in scheduler else base_lr
    weight_decay = cfg.training['weight_decay']

    if cfg.training['optim'] == 'AdamW':
        optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer

def get_scheduler(cfg, optimizer):
    if cfg.training['scheduler'] == 'MultiStepLR':
        gamma = cfg.training['gamma']
        milestones = cfg.training['milestones']
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    elif cfg.training['scheduler'] == 'CosineAnnealingWarmUpRestarts':
        T_0 = cfg.training['T_0']
        T_up = cfg.training['T_up']
        lr_max = cfg.training['lr_max']
        gamma = cfg.training['gamma']
        lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=T_0, T_up=T_up, eta_max=lr_max, gamma=gamma)

    else:
        gamma = 1
        milestones = [cfg.training['epochs']]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    return lr_scheduler


class CosineAnnealingWarmUpRestarts(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, T_up=0, eta_max=0.1, gamma=1., last_epoch=-1):
        """
        T0      : Time period
        T_multi : Time period multiplier
        eta_max : The maximum value of learning rate
        T_up    : The number of warm up epochs
        gamma   : The scale factor of eta_max for every period
        """
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))

        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cosine_period = self.T_0 - self.T_up if self.T_mult == 1 else self.T_i - self.T_up
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur > cosine_period:
            return [(self.eta_max * self.gamma - base_lr) * (self.T_cur - cosine_period) / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2 for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
