#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataloader import default_collate
import dataset
import naggn
import dragn
import transformer
import sinet
import util
import train


def train_smallnet_stratify_pj(s=2, k=10):
    cfg = util.default_cfg()
    cfg = train._config_stratify_pj_dataset(cfg, s, k)

    model = nn.DataParallel(sinet.SmallNet(k=k).cuda())
    cfg['model'] = 'smallnet_pj_k%d' % (k)
    cfg['model_dir'] = 'modeldir/stage%d/smallnet_pj_k%d' % (s, k)
    cfg = train._train_config_pj(model, cfg)

    train.run_train(model, cfg)


def train_smallnet_stratify_si(s=2, k=10):
    cfg = util.default_cfg()
    cfg = train._config_stratify_si_dataset(cfg, s, k)
    import loss
    cfg['criterion'] = loss.FECLoss(alpha=64)

    model = nn.DataParallel(sinet.SmallNet(k=k).cuda())
    cfg['model'] = 'smallnet_si_k%d_fec1' % (k)
    cfg['model_dir'] = 'modeldir/stage%d/smallnet_si_k%d_fec1' % (s, k)
    cfg = train._train_config_si(model, cfg)
    cfg['scheduler'] = False
    cfg['lr'] = 0.0001

    train.run_train(model, cfg)


def train_smallnet_stratify_si_fech(s=2, k=10):
    cfg = util.default_cfg()
    cfg = train._config_stratify_si_dataset(cfg, s, k)
    import loss
    cfg['criterion'] = loss.FECLoss(alpha=32)

    model = nn.DataParallel(sinet.SmallNet(k=k).cuda())
    cfg['model'] = 'smallnet_si_k%d_fec0.5' % (k)
    cfg['model_dir'] = 'modeldir/stage%d/smallnet_si_k%d_fec0.5' % (s, k)
    cfg = train._train_config_si(model, cfg)
    cfg['scheduler'] = False
    cfg['lr'] = 0.0001
    cfg['epochs'] = 1000

    train.run_train(model, cfg)


def train_smallnet_stratify_si_fecq(s=2, k=10):
    cfg = util.default_cfg()
    cfg = train._config_stratify_si_dataset(cfg, s, k)
    import loss
    cfg['criterion'] = loss.FECLoss(alpha=16)

    model = nn.DataParallel(sinet.SmallNet(k=k).cuda())
    cfg['model'] = 'smallnet_si_k%d_fec0.25' % (k)
    cfg['model_dir'] = 'modeldir/stage%d/smallnet_si_k%d_fec0.25' % (s, k)
    cfg = train._train_config_si(model, cfg)
    cfg['scheduler'] = False
    cfg['lr'] = 0.0001
    cfg['epochs'] = 1000

    train.run_train(model, cfg)


def train_tinynet_stratify_si(s=2, k=10):
    cfg = util.default_cfg()
    cfg = train._config_stratify_si_dataset(cfg, s, k)

    from loss import FECLoss
    cfg['criterion'] = FECLoss(alpha=32)
    cfg['model'] = 'tinynet_si_k%d_fec0.5' % (k)
    cfg['model_dir'] = 'modeldir/stage%d/tinynet_si_k%d_fec0.5' % (s, k)
    cfg['collate'] = default_collate
    cfg['instance'] = train._train_si

    model = nn.DataParallel(sinet.TinyNet(k=k).cuda())

    cfg = train._train_config_si(model, cfg)

    train.run_train(model, cfg)


def train_senet_si(s=2, k=10, val_index=4):
    cfg = util.default_cfg()
    cfg = train._config_si_dataset(cfg, s, k)

    cfg['model'] = 'senet_si_k%d_val%d' % (k, val_index)
    cfg['model_dir'] = 'modeldir/stage%d/senet_si_k%d_val%d' % (
        s, k, val_index)

    model = nn.DataParallel(sinet.FlySENet(k=k).cuda())
    cfg = train._train_config_si(model, cfg)

    train.run_train(model, cfg)


def train_resnet_si(s=2, k=10, val_index=4):
    cfg = util.default_cfg()
    cfg = train._config_si_dataset(cfg, s, k)

    cfg['model'] = 'resnet_si_k%d_val%d' % (k, val_index)
    cfg['model_dir'] = 'modeldir/stage%d/resnet_si_k%d_val%d' % (
        s, k, val_index)

    model = nn.DataParallel(sinet.SiNet(nblock=4, k=k).cuda())
    cfg = train._train_config_si(model, cfg)

    train.run_train(model, cfg)


def train_resnet_pj(s=2, k=10):
    cfg = util.default_cfg()
    cfg = train._config_pj_dataset(cfg, s, k)

    cfg['model'] = 'resnet_pj_k%d' % (k)
    cfg['model_dir'] = 'modeldir/stage%d/resnet_pj_k%d' % (
        s, k)

    model = nn.DataParallel(sinet.SiNet(nblock=4, k=k).cuda())
    cfg = train._train_config_pj(model, cfg)

    train.run_train(model, cfg)


def train_naggn(s=2):
    train_dataset = dataset.DrosophilaDataset(mode='train', stage=s)
    val_dataset = dataset.DrosophilaDataset(mode='val', stage=s)
    test_dataset = dataset.DrosophilaDataset(mode='test', stage=s)

    cfg = util.default_cfg()
    cfg['train'] = train_dataset
    cfg['val'] = val_dataset
    cfg['test'] = test_dataset
    cfg['batch'] = 64
    cfg['lr'] = 0.000001
    cfg['model'] = 'naggn_l1'
    cfg['model_dir'] = 'modeldir/stage%d/naggn_l1' % s
    cfg['collate'] = dataset.fly_collate_fn
    cfg['instance'] = train._train_mi

    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    model = nn.DataParallel(naggn.NAggN(agg='l1').cuda())
    if os.path.exists(model_pth):
        ckp = torch.load(model_pth)
        model.load_state_dict(ckp['model'])
        cfg['step'] = ckp['epoch'] + 1
        print("load pretrained model", model_pth, "start epoch:", cfg['step'])

    train.run_train(model, cfg)


def train_prefv_naggn(s=2):
    '''load fvextractor from pretrained resnet_si'''
    train_dataset = dataset.DrosophilaDataset(mode='train', stage=s)
    val_dataset = dataset.DrosophilaDataset(mode='val', stage=s)
    test_dataset = dataset.DrosophilaDataset(mode='test', stage=s)

    cfg = util.default_cfg()
    cfg['train'] = train_dataset
    cfg['val'] = val_dataset
    cfg['test'] = test_dataset
    cfg['batch'] = 32
    cfg['lr'] = 0.0001
    cfg['model'] = 'prefv_naggn_l2'
    cfg['model_dir'] = 'modeldir/stage%d/prefv_naggn_l2' % s
    cfg['collate'] = dataset.fly_collate_fn
    cfg['instance'] = train._train_mi

    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    model = nn.DataParallel(naggn.NAggN(agg='l2').cuda())
    if os.path.exists(model_pth):
        ckp = torch.load(model_pth)
        model.load_state_dict(ckp['model'])
        cfg['step'] = ckp['epoch'] + 1
        print("load pretrained model", model_pth, "start epoch:", cfg['step'])
    else:
        fv_model_dir = 'modeldir/stage%d/resnet_si' % s
        fv_model_pth = os.path.join(fv_model_dir, 'model.pth')
        ckp = torch.load(fv_model_pth)
        model.state_dict().update(ckp['model'])
        print("load fvextractor from pretrained resnet_si")

    # for p in model.module.fvextractor.parameters():
    #     p.require_grad = False

    train.run_train(model, cfg)


def train_dragn(s=2):
    train_dataset = dataset.DrosophilaDataset(mode='train', stage=s)
    val_dataset = dataset.DrosophilaDataset(mode='val', stage=s)
    test_dataset = dataset.DrosophilaDataset(mode='test', stage=s)

    cfg = util.default_cfg()
    cfg['train'] = train_dataset
    cfg['val'] = val_dataset
    cfg['test'] = test_dataset
    cfg['batch'] = 2
    cfg['lr'] = 0.0001
    cfg['model'] = 'dragn'
    cfg['model_dir'] = 'modeldir/stage%d/dragn' % s
    cfg['collate'] = dataset.fly_collate_fn
    cfg['instance'] = train._train_si

    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    model = nn.DataParallel(dragn.DRAGN().cuda())
    if os.path.exists(model_pth):
        print("load pretrained model", model_pth)
        model.load_state_dict(torch.load(model_pth))
    train.run_train(model, cfg)


def train_transformer(s=2):
    train_dataset = dataset.DrosophilaDataset(mode='train', stage=s)
    val_dataset = dataset.DrosophilaDataset(mode='val', stage=s)
    test_dataset = dataset.DrosophilaDataset(mode='test', stage=s)

    cfg = util.default_cfg()
    cfg['train'] = train_dataset
    cfg['val'] = val_dataset
    cfg['test'] = test_dataset
    cfg['batch'] = 32
    cfg['lr'] = 0.00001
    cfg['model'] = 'transformer'
    cfg['model_dir'] = 'modeldir/stage%d/transformer' % s
    cfg['collate'] = dataset.fly_collate_fn
    cfg['instance'] = train._train_mi

    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    model = nn.DataParallel(transformer.E2ETransformer().cuda())
    if os.path.exists(model_pth):
        ckp = torch.load(model_pth)
        model.load_state_dict(ckp['model'])
        cfg['step'] = ckp['epoch'] + 1
        print("load pretrained model", model_pth, "start epoch:", cfg['step'])

    train.run_train(model, cfg)


def run_kfold_test(k=10):
    for s in [6, 5, 4, 3, 2]:
        test_dataset = dataset.SIDataset(mode='test', stage=s, k=k)

        s_dir = 'modeldir/stage%d' % s
        s_score = []
        s_label = []
        for val_index in [4, 3, 2, 1, 0]:
            m_dir = '%s/resnet_si_k%d_val%d' % (s_dir, k, val_index)

            model_pth = os.path.join(m_dir, 'model.pth')

            model = nn.DataParallel(sinet.SiNet(nblock=4, k=k).cuda())
            ckp = torch.load(model_pth)
            model.load_state_dict(ckp['model'])

            cfg = util.default_cfg()
            cfg['test'] = test_dataset
            cfg['batch'] = 128
            cfg['collate'] = default_collate
            cfg['instance'] = train._train_si
            cfg['model'] = m_dir

            np_score, np_label = train.run_test_score(model, cfg)
            s_score.append(np_score)
            s_label.append(np_label)

        m_score = np.mean(np.stack(s_score, axis=0), axis=0)
        print("m_score", m_score.shape, 'np_label', np_label.shape)
        np_pd = (m_score > 0.5).astype(np.int)

        mean_dir = '%s/resnet_si_k%d_mean' % (s_dir, k)
        if not os.path.exists(mean_dir):
            os.mkdir(mean_dir)
        pth = os.path.join(mean_dir, 'metrics.csv')
        util.write_metrics(pth, np_label, np_pd, np_score)


def sequence_train_stratify():
    import multiprocessing as mp
    for s in [2, 3, 4, 5, 6]:
        p = mp.Process(target=train_smallnet_stratify_pj, args=(s, 10))
        p.start()
        p.join()

    # for s in [2, 3, 4, 5, 6]:
    #     p = mp.Process(target=train_resnet_stratify_si, args=(s, 10))
    #     p.start()
    #     p.join()


def sequence_train_stratify_si():
    import multiprocessing as mp
    for s in [2, 3, 4, 5, 6]:
        p = mp.Process(target=train_smallnet_stratify_si_fech, args=(s, 10))
        p.start()
        p.join()

    for s in [2, 3, 4, 5, 6]:
        p = mp.Process(target=train_smallnet_stratify_si_fecq, args=(s, 10))
        p.start()
        p.join()


def sequence_train_kfold():
    import multiprocessing as mp
    for s in [6, 5, 4, 3, 2]:
        for val_index in [4, 3, 2, 1, 0]:
            p = mp.Process(target=train_resnet_si, args=(s, 10, val_index))
            p.start()
            p.join()

if __name__ == "__main__":
    # train_transformer(s=2)
    # train_naggn(s=2)
    # train_prefv_naggn(s=6)
    # train_resnet_si(s=2, k=10, val_index=4)
    # train_resnet_pj(s=2)
    # sequence_train()
    # run_kfold_test()
    # train_senet_si(s=2, k=10, val_index=4)
    # train_tinynet_stratify_si(s=2, k=10)
    # train_smallnet_stratify_pj(s=6, k=10)
    # sequence_train_stratify()
    # train_smallnet_stratify_si(s=2, k=10)
    sequence_train_stratify_si()
