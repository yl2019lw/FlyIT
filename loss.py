#!/usr/bin/env python
# -*- coding: utf-8 -*-

# AutoLoss copy from my HPA paper
# FocalBCELoss & FocalMSELoss copy from LTG

import torch
import torch.nn as nn
import torch.nn.functional as F

epsilon = 1e-8


class RectifyLoss(nn.Module):
    '''RectifyLoss'''

    def __init__(self, gamma=1, thr=0.5):
        super(RectifyLoss, self).__init__()
        self.gamma = gamma
        self.thr = thr

    def agree_mask(self, p, y):
        '''return 0-1 agree mask, p > 0.5 for y = 1, p < 0.5 for y = 0'''
        sign = (p - self.thr) * (y - self.thr)
        return torch.sigmoid(1e8 * sign)

    def mask_tp(self, p, y):
        '''tp is y == 1 and p agree with y'''
        return y * self.agree

    def mask_fp(self, p, y):
        '''fp is y == 0 and p not agree with y'''
        return (1 - y) * (1 - self.agree)

    def mask_tn(self, p, y):
        '''tn is y == 0 and p agree with y'''
        return (1 - y) * self.agree

    def mask_fn(self, p, y):
        '''fn is y == 1 and p not agree with y'''
        return y * (1 - self.agree)

    def forward(self, p, y):
        pass


class FECLoss(nn.Module):
    '''auto weighted loss, focus on error correction'''
    def __init__(self, alpha=100, gamma=1,
                 reduction='mean', thr=0.5):
        '''
        alpha controls weights between bce & penalty.
        gamma controls penalty level.
        p > thr as positive, otherwise negative.
        '''
        super(FECLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.thr = thr

    def agree_mask(self, p, y):
        '''return 0-1 agree mask, p > 0.5 for y = 1, p < 0.5 for y = 0'''
        sign = (p - self.thr) * (y - self.thr)
        return torch.sigmoid(1e8 * sign)

    def mask_tp(self, p, y):
        '''tp is y == 1 and p agree with y'''
        return y * self.agree

    def mask_fp(self, p, y):
        '''fp is y == 0 and p not agree with y'''
        return (1 - y) * (1 - self.agree)

    def mask_tn(self, p, y):
        '''tn is y == 0 and p agree with y'''
        return (1 - y) * self.agree

    def mask_fn(self, p, y):
        '''fn is y == 1 and p not agree with y'''
        return y * (1 - self.agree)

    def forward(self, p, y):
        oloss = F.binary_cross_entropy(p, y, reduction='none')

        self.agree = self.agree_mask(p, y)
        nsample, nlabel = y.shape

        tp_ind = self.mask_tp(p, y)
        fp_ind = self.mask_fp(p, y)
        tn_ind = self.mask_tn(p, y)
        fn_ind = self.mask_fn(p, y)

        tp = torch.sum(tp_ind, dim=0)
        fp = torch.sum(fp_ind, dim=0)
        tn = torch.sum(tn_ind, dim=0)
        fn = torch.sum(fn_ind, dim=0)

        fp_coef = fp / (tn + 1)
        fn_coef = fn / (tp + 1)
        fp_w = fp_coef * fp_ind
        fn_w = fn_coef * fn_ind

        penalty = fp_w ** self.gamma + fn_w ** self.gamma
        weights = 1 + self.alpha * penalty / nsample

        w_loss = weights * oloss

        if self.reduction == 'none':
            return w_loss
        else:
            return torch.mean(w_loss)


class F1Loss(nn.Module):
    '''add f1 term to loss'''

    def __init__(self, bce=False, factor=1.0, thr=0.5):
        super(F1Loss, self).__init__()
        self.bce = bce
        self.factor = factor
        self.thr = thr

    def forward(self, p, y):
        floss = self.compute_floss(p, y).float()
        if self.bce:
            oloss = F.binary_cross_entropy(p, y, reduction='elementwise_mean')
            floss = oloss + floss * self.factor
        return floss

    def agree_mask(self, p, y):
        '''return 0-1 agree mask, p > 0.5 for y = 1, p < 0.5 for y = 0'''
        sign = (p - self.thr) * (y - self.thr)
        return torch.sigmoid(1e8 * sign)

    def mask_tp(self, p, y):
        '''tp is y == 1 and p agree with y'''
        return y * self.agree

    def mask_fp(self, p, y):
        '''fp is y == 0 and p not agree with y'''
        return (1 - y) * (1 - self.agree)

    def mask_tn(self, p, y):
        '''tn is y == 0 and p agree with y'''
        return (1 - y) * self.agree

    def mask_fn(self, p, y):
        '''fn is y == 1 and p not agree with y'''
        return y * (1 - self.agree)

    def compute_floss(self, p, y):
        self.agree = self.agree_mask(p, y)

        tp_ind = self.mask_tp(p, y)
        fp_ind = self.mask_fp(p, y)
        fn_ind = self.mask_fn(p, y)

        tp = torch.sum(tp_ind, dim=0)
        fp = torch.sum(fp_ind, dim=0)
        fn = torch.sum(fn_ind, dim=0)

        f1 = tp / (tp + 0.5 * fp + 0.5 * fn + epsilon)
        ma_f1 = torch.mean(f1)

        return 1 - ma_f1


class FocalMSELoss(nn.Module):
    """
    this criterion is a implementation of Focal Loss,
    Loss(y,p)=alpha*(1-p)^gamma*(y-p)^2 when y=1
             =(1-alpha)*p^gamma*(y-p)^2
    Args:
        alpha(float,double): a scalar factor for this criterion
        gamma(float,double): reduces the relative loss for well-classified
            examples, and putting more focus on hard, misclassified examples
        size_average(bool): By default, the losses are averaged over each
            minibatch. However, if the field size_average is set to False,
            then the losses are instead summed for each minibatch
    """

    def __init__(self, alpha, gamma, size_average):
        super(FocalMSELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        MseLoss = torch.pow(targets - inputs, 2)
        chaZhi = torch.abs(targets - inputs)
        MseLoss = torch.mul(MseLoss, torch.pow(chaZhi, self.gamma))

        for i in range(N):
            for j in range(C):
                if targets[i][j] == 1:
                    MseLoss[i][j] = MseLoss[i][j]*self.alpha[j]
        if self.size_average:
            loss = MseLoss.mean()
        else:
            loss = MseLoss.sum()
        return loss


class FocalBCELoss(nn.Module):

    def __init__(self, alpha, gamma, size_average):
        super(FocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)

        PosLoss = torch.mul(torch.log(inputs), torch.pow(1-inputs, self.gamma))
        LeftLoss = torch.mul(targets, PosLoss)
        for i in range(N):
            for j in range(C):
                LeftLoss[i][j] = LeftLoss[i][j] * self.alpha[j]

        NegLoss = torch.mul(torch.log(1-inputs), torch.pow(inputs, self.gamma))
        RightLoss = torch.mul(1-targets, NegLoss)

        Loss = -1*(RightLoss + LeftLoss)

        if self.size_average:
            loss = Loss.mean()
        else:
            loss = Loss.sum()
        return loss
