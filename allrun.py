#!/usr/bin/env python
# -*- coding: utf-8 -*-

# why allrun, because treat gene + stage as sid

import os
import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate
import alldataset
import sinet
import util
import train


def _allrun_config_si(k=10):
    cfg = util.default_cfg()

    train_dataset = alldataset.SIDataset(mode='train', k=k)
    val_dataset = alldataset.SIDataset(mode='val', k=k)
    test_dataset = alldataset.SIDataset(mode='test', k=k)

    cfg['train'] = train_dataset
    cfg['val'] = val_dataset
    cfg['test'] = test_dataset

    cfg['batch'] = 64
    cfg['epochs'] = 500
    cfg['scheduler'] = True
    cfg['decay'] = 0.01
    cfg['lr'] = 0.0001
    cfg['patience'] = 20
    cfg['collate'] = default_collate
    cfg['instance'] = train._train_si

    return cfg


def _allrun_config_pj(k=10):
    cfg = util.default_cfg()

    train_dataset = alldataset.PJDataset(mode='train', k=k)
    val_dataset = alldataset.PJDataset(mode='val', k=k)
    test_dataset = alldataset.PJDataset(mode='test', k=k)

    cfg['train'] = train_dataset
    cfg['val'] = val_dataset
    cfg['test'] = test_dataset

    cfg['batch'] = 32
    cfg['epochs'] = 500
    cfg['scheduler'] = True
    cfg['decay'] = 0.01
    cfg['lr'] = 0.0001
    cfg['patience'] = 20
    cfg['collate'] = default_collate
    cfg['instance'] = train. _train_si

    return cfg


def train_resnet_si(k=10):
    cfg = _allrun_config_si(k)
    # from loss import FECLoss
    # cfg['criterion'] = FECLoss(alpha=48)

    model = nn.DataParallel(sinet.SiNet(nblock=4, k=k).cuda())
    cfg['model'] = 'resnet18b4_si_k%d' % (k)
    cfg['model_dir'] = 'modeldir/stage_all/resnet18b4_si_k%d' % (k)
    cfg['lr'] = 0.0001

    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    if os.path.exists(model_pth):
        ckp = torch.load(model_pth)
        model.load_state_dict(ckp['model'])
        cfg['step'] = ckp['epoch'] + 1
        print("load pretrained model", model_pth, "start epoch:", cfg['step'])

    train.run_train(model, cfg)


def train_smallnet_si(k=10):
    cfg = _allrun_config_si(k)
    # from loss import FECLoss
    # cfg['criterion'] = FECLoss(alpha=48)

    model = nn.DataParallel(sinet.SmallNet(k=k).cuda())
    cfg['model'] = 'smallnet_si_k%d' % (k)
    cfg['model_dir'] = 'modeldir/stage_all/smallnet_si_k%d' % (k)

    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    if os.path.exists(model_pth):
        ckp = torch.load(model_pth)
        model.load_state_dict(ckp['model'])
        cfg['step'] = ckp['epoch'] + 1
        print("load pretrained model", model_pth, "start epoch:", cfg['step'])

    train.run_train(model, cfg)


def train_resnet34_si(k=10):
    cfg = _allrun_config_si(k)

    model = nn.DataParallel(sinet.Resnet34(nblock=4, k=k).cuda())
    cfg['model'] = 'resnet34b4_si_k%d' % (k)
    cfg['model_dir'] = 'modeldir/stage_all/resnet34b4_si_k%d' % (k)
    cfg['lr'] = 0.0001
    cfg['scheduler'] = True

    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    if os.path.exists(model_pth):
        ckp = torch.load(model_pth)
        model.load_state_dict(ckp['model'])
        cfg['step'] = ckp['epoch'] + 1
        print("load pretrained model", model_pth, "start epoch:", cfg['step'])

    train.run_train(model, cfg)


def train_densenet121_si(k=10):
    cfg = _allrun_config_si(k)

    model = nn.DataParallel(sinet.Densenet121(k=k).cuda())
    cfg['model'] = 'densenet121_si_k%d' % (k)
    cfg['model_dir'] = 'modeldir/stage_all/densenet121_si_k%d' % (k)
    cfg['lr'] = 0.0001
    cfg['scheduler'] = True

    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    if os.path.exists(model_pth):
        ckp = torch.load(model_pth)
        model.load_state_dict(ckp['model'])
        cfg['step'] = ckp['epoch'] + 1
        print("load pretrained model", model_pth, "start epoch:", cfg['step'])

    train.run_train(model, cfg)


def train_resnet_pj(k=10):
    cfg = _allrun_config_pj(k)
    # from loss import FECLoss
    # cfg['criterion'] = FECLoss(alpha=48)

    model = nn.DataParallel(sinet.SiNet(nblock=4, k=k).cuda())
    cfg['model'] = 'resnet18b4_pj_k%d' % (k)
    cfg['model_dir'] = 'modeldir/stage_all/resnet18b4_pj_k%d' % (k)

    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    if os.path.exists(model_pth):
        ckp = torch.load(model_pth)
        model.load_state_dict(ckp['model'])
        cfg['step'] = ckp['epoch'] + 1
        print("load pretrained model", model_pth, "start epoch:", cfg['step'])

    train.run_train(model, cfg)


def train_resnet34_pj(k=10):
    cfg = _allrun_config_pj(k)
    # from loss import FECLoss
    # cfg['criterion'] = FECLoss(alpha=48)

    model = nn.DataParallel(sinet.Resnet34(nblock=4, k=k).cuda())
    cfg['model'] = 'resnet34b4_pj_k%d' % (k)
    cfg['model_dir'] = 'modeldir/stage_all/resnet34b4_pj_k%d' % (k)
    cfg['lr'] = 0.0001
    cfg['scheduler'] = True

    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    if os.path.exists(model_pth):
        ckp = torch.load(model_pth)
        model.load_state_dict(ckp['model'])
        cfg['step'] = ckp['epoch'] + 1
        print("load pretrained model", model_pth, "start epoch:", cfg['step'])

    train.run_train(model, cfg)


def train_resnet50_pj(k=10):
    cfg = _allrun_config_pj(k)
    # from loss import FECLoss
    # cfg['criterion'] = FECLoss(alpha=48)

    model = nn.DataParallel(sinet.Resnet50(nblock=4, k=k).cuda())
    cfg['model'] = 'resnet50b4_pj_k%d' % (k)
    cfg['model_dir'] = 'modeldir/stage_all/resnet50b4_pj_k%d' % (k)
    cfg['lr'] = 0.0001
    cfg['scheduler'] = True

    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    if os.path.exists(model_pth):
        ckp = torch.load(model_pth)
        model.load_state_dict(ckp['model'])
        cfg['step'] = ckp['epoch'] + 1
        print("load pretrained model", model_pth, "start epoch:", cfg['step'])

    train.run_train(model, cfg)


if __name__ == "__main__":
    train_resnet_si(k=20)
    # train_smallnet_si()
    # train_resnet34_si()
    # train_densenet121_si()

    # train_resnet_pj()
    # train_resnet34_pj()
    # train_resnet50_pj()
