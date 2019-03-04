#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import psutil
import npmetrics
from sklearn import metrics

NUM_CLASSES = 10


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


def sklearn_auc_macro(gt, score):
    return metrics.roc_auc_score(gt, score, average='macro')


def sklearn_auc_micro(gt, score):
    return metrics.roc_auc_score(gt, score, average='micro')


def sklearn_f1_macro(gt, predict):
    return metrics.f1_score(gt, predict, average='macro')


def sklearn_f1_micro(gt, predict):
    return metrics.f1_score(gt, predict, average='micro')


def torch_metrics(gt, predict, writer, step, mode="val", score=None):

    sk_auc_macro = sklearn_auc_macro(gt, score)
    # sk_auc_micro = sklearn_auc_micro(gt, score)
    sk_f1_macro = sklearn_f1_macro(gt, predict)
    sk_f1_micro = sklearn_f1_micro(gt, predict)

    lab_f1_macro = npmetrics.label_f1_macro(gt, predict)
    lab_f1_micro = npmetrics.label_f1_micro(gt, predict)
    lab_sensitivity = npmetrics.label_sensitivity(gt, predict)
    lab_specificity = npmetrics.label_specificity(gt, predict)

    writer.add_scalar("%s auc macro sk" % mode, sk_auc_macro, step)
    # writer.add_scalar("%s auc micro sk" % mode, sk_auc_micro, step)
    writer.add_scalar("%s label f1 macro sk" % mode, sk_f1_macro, step)
    writer.add_scalar("%s label f1 micro sk" % mode, sk_f1_micro, step)

    writer.add_scalar("%s label f1 macro" % mode, lab_f1_macro, step)
    writer.add_scalar("%s label f1 micro" % mode, lab_f1_micro, step)
    writer.add_scalar("%s label sensitivity" % mode, lab_sensitivity, step)
    writer.add_scalar("%s label specificity" % mode, lab_specificity, step)

    sl_acc = npmetrics.single_label_accuracy(gt, predict)
    sl_precision = npmetrics.single_label_precision(gt, predict)
    sl_recall = npmetrics.single_label_recall(gt, predict)
    for i in range(NUM_CLASSES):
        writer.add_scalar("%s sl_%d_acc" % (mode, i),
                          sl_acc[i], step)
        writer.add_scalar("%s sl_%d_precision" % (mode, i),
                          sl_precision[i], step)
        writer.add_scalar("%s sl_%d_recall" % (mode, i),
                          sl_recall[i], step)
    return lab_f1_macro


def write_metrics(pth, gt, predict, score=None):
    '''write test metrics'''
    sk_auc_macro = sklearn_auc_macro(gt, score)
    sk_f1_macro = sklearn_f1_macro(gt, predict)
    sk_f1_micro = sklearn_f1_micro(gt, predict)
    lab_sensitivity = npmetrics.label_sensitivity(gt, predict)
    lab_specificity = npmetrics.label_specificity(gt, predict)

    with open(pth, 'w') as f:
        f.write('auc:%.4f\n' % sk_auc_macro)
        f.write('f1_macro:%.4f\n' % sk_f1_macro)
        f.write('f1_micro:%.4f\n' % sk_f1_micro)
        f.write('sensitivity:%.4f\n' % lab_sensitivity)
        f.write('specificity:%.4f\n' % lab_specificity)


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
