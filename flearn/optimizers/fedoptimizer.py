from torch.optim import Optimizer
import torch


class MySGD(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(MySGD, self).__init__(params, defaults)

    def step(self, closure=None, hyper_learning_rate=0):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if (hyper_learning_rate != 0):
                    p.data.add_(-hyper_learning_rate, d_p)
                else:
                    p.data.add_(-group['lr'], d_p)
        return loss


class FEDLOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, hyper_lr=0.01, L=0.1):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, hyper_lr=hyper_lr, L=L)
        super(FEDLOptimizer, self).__init__(params, defaults)

    def step(self, server_grads, pre_grads, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        for group in self.param_groups:
            for p, server_grad, pre_grad in zip(group['params'], server_grads, pre_grads):
                if (server_grad.grad != None and pre_grad.grad != None):
                    p.data = p.data - group['lr'] * (
                                p.grad.data + group['hyper_lr'] * server_grad.grad.data - pre_grad.grad.data)
                else:
                    p.data = p.data - group['lr'] * p.grad.data
        return loss


class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, L=0.1, mu=0.001):
        # self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, L=L, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    def step(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                p.data = p.data - group['lr'] * (
                            p.grad.data + group['L'] * (p.data - localweight.data) + group['mu'] * p.data)
        return group['params'], loss

    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                p.data = localweight.data
        # return  p.data
        return group['params']


class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)
        pass

    def step(self, server_controls, client_controls, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        for group, c, ci in zip(self.param_groups, server_controls, client_controls):
            p = group['params'][0]
            if p.grad is None:
                continue
            d_p = p.grad.data + c.data - ci.data
            p.data = p.data - d_p.data * group['lr']
        # for group in self.param_groups:
        #     for p, c, ci in zip(group['params'], server_controls, client_controls):
        #         if p.grad is None:
        #             continue
        #         d_p = p.grad.data + c.data - ci.data
        #         p.data = p.data - d_p.data * group['lr']
        return loss
