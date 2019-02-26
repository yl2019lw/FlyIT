#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import psutil
from sklearn import metrics


def default_cfg():
    cfg = {}
    cfg['epochs'] = 10000
    cfg['batch'] = 96
    cfg['nworker'] = psutil.cpu_count()
    cfg['criterion'] = torch.nn.BCELoss()
    cfg['lr'] = 0.001
    cfg['decay'] = 0.001
    cfg['scheduler'] = False
    cfg['factor'] = 0.1
    cfg['patience'] = 10
    return cfg


def label_auc(gt, predict):
    return metrics.roc_auc_score(gt, predict, average='macro')


def label_f1_macro(gt, predict):
    return metrics.f1_score(gt, predict, average='macro')


def label_f1_micro(gt, predict):
    return metrics.f1_score(gt, predict, average='micro')


def label_sensitivity(gt, predict):
    pass


def label_specificity(gt, predict):
    pass


def torch_metrics(gt, predict, writer, step, mode="val"):
    lab_f1_macro = label_f1_macro(gt, predict)
    lab_f1_micro = label_f1_micro(gt, predict)

    writer.add_scalar("%s label f1 macro" % mode, lab_f1_macro, step)
    writer.add_scalar("%s label f1 micro" % mode, lab_f1_micro, step)

    return lab_f1_macro


def threshold_tensor_batch(predict, base=0.5):
    '''make sure at least one label for batch'''
    p_max = torch.max(predict, dim=1)[0]
    pivot = torch.cuda.FloatTensor([base]).expand_as(p_max)
    threshold = torch.min(p_max, pivot)
    pd_threshold = torch.ge(predict, threshold.unsqueeze(dim=1))
    return pd_threshold


def threshold_predict(predict, base=0.5):
    '''make sure at least one label for one example'''
    p_max = torch.max(predict)
    pivot = torch.cuda.FloatTensor([base])
    threshold = torch.min(p_max, pivot)
    pd_threshold = torch.ge(predict, threshold)
    return pd_threshold
