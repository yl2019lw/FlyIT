#!/usr/bin/env python
# -*- coding: utf-8 -*-

# treat gene + stage as sid, use feature aggregation

import os
import torch
import torch.nn as nn
import alldataset
import transformer
import dragn
import naggn
import util
import train


def _allrun_config_agg(k=10):
    cfg = util.default_cfg()

    train_dataset = alldataset.AggDataset(mode='train', k=k)
    val_dataset = alldataset.AggDataset(mode='val', k=k)
    test_dataset = alldataset.AggDataset(mode='test', k=k)

    cfg['train'] = train_dataset
    cfg['val'] = val_dataset
    cfg['test'] = test_dataset

    cfg['batch'] = 64
    cfg['epochs'] = 500
    cfg['scheduler'] = True
    cfg['decay'] = 0.01
    cfg['lr'] = 0.0001
    cfg['patience'] = 20
    cfg['collate'] = alldataset.fly_collate_fn
    cfg['instance'] = train._train_mi

    return cfg


def train_transformer_agg(k=10):
    cfg = _allrun_config_agg(k)
    cfg['model'] = 'transformer'
    cfg['model_dir'] = 'modeldir/agg_stage_all/transformer_k%d' % k
    cfg['lr'] = 0.0001
    cfg['nworker'] = 8
    cfg['batch'] = 16

    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    model = nn.DataParallel(transformer.E2ETransformer().cuda())
    if os.path.exists(model_pth):
        ckp = torch.load(model_pth)
        model.load_state_dict(ckp['model'])
        cfg['step'] = ckp['epoch'] + 1
        print("load pretrained model", model_pth, "start epoch:", cfg['step'])

    train.run_train(model, cfg)


def train_naggn_agg(k=10):
    cfg = _allrun_config_agg(k)
    cfg['model'] = 'naggn-l1'
    cfg['model_dir'] = 'modeldir/agg_stage_all/naggn-l1_k%d' % k
    cfg['lr'] = 0.0001
    cfg['nworker'] = 8

    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    model = nn.DataParallel(naggn.NAggN(agg='l1').cuda())
    if os.path.exists(model_pth):
        ckp = torch.load(model_pth)
        model.load_state_dict(ckp['model'])
        cfg['step'] = ckp['epoch'] + 1
        print("load pretrained model", model_pth, "start epoch:", cfg['step'])

    train.run_train(model, cfg)


def train_post_dragn_agg(k=10):
    cfg = _allrun_config_agg(k)
    cfg['model'] = 'post_dragn'
    cfg['model_dir'] = 'modeldir/agg_stage_all/post_dragn_k%d-1layer' % k
    cfg['lr'] = 0.0001
    cfg['batch'] = 32

    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    model = nn.DataParallel(dragn.PostDRAGN(k, agglevel=1).cuda())
    if os.path.exists(model_pth):
        ckp = torch.load(model_pth)
        model.load_state_dict(ckp['model'])
        cfg['step'] = ckp['epoch'] + 1
        print("load pretrained model", model_pth, "start epoch:", cfg['step'])

    train.run_train(model, cfg)


def train_pre_dragn_agg(k=10):
    cfg = _allrun_config_agg(k)
    cfg['model'] = 'pre_dragn'
    cfg['model_dir'] = 'modeldir/agg_stage_all/pre_dragn_k%d-1layer' % k
    cfg['lr'] = 0.0001
    cfg['batch'] = 32

    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    model = nn.DataParallel(dragn.PreDRAGN(k, agglevel=1).cuda())
    if os.path.exists(model_pth):
        ckp = torch.load(model_pth)
        model.load_state_dict(ckp['model'])
        cfg['step'] = ckp['epoch'] + 1
        print("load pretrained model", model_pth, "start epoch:", cfg['step'])

    train.run_train(model, cfg)


if __name__ == "__main__":
    # train_transformer_agg()
    # train_naggn_agg()
    train_post_dragn_agg(k=10)
    # train_pre_dragn_agg()
