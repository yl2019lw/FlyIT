#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import tensorboardX
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import dataset
import naggn
import dragn
import util
import extractor

FV_DIM = 512


class SiNet(nn.Module):

    def __init__(self, k=10):
        super(SiNet, self).__init__()
        self.fvextractor = extractor.Resnet()
        self.proj = nn.Linear(FV_DIM, k)

    def forward(self, x):
        fv = self.fvextractor(x)
        return torch.sigmoid(self.proj(fv))


def train_resnet_si(s=2):
    train_dataset = dataset.SIDataset(mode='train', stage=s)
    val_dataset = dataset.SIDataset(mode='val', stage=s)
    test_dataset = dataset.SIDataset(mode='test', stage=s)

    cfg = util.default_cfg()
    cfg['train'] = train_dataset
    cfg['val'] = val_dataset
    cfg['test'] = test_dataset
    cfg['batch'] = 64
    cfg['lr'] = 0.0001
    cfg['model'] = 'resnet_si'
    cfg['model_dir'] = 'modeldir/stage%d/resnet_si' % s
    cfg['collate'] = default_collate
    cfg['instance'] = _train_si

    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    model = nn.DataParallel(SiNet().cuda())
    if os.path.exists(model_pth):
        print("load pretrained model", model_pth)
        model.load_state_dict(torch.load(model_pth))

    run_train(model, cfg)


def train_resnet_pj(s=2):
    train_dataset = dataset.PJDataset(mode='train', stage=s)
    val_dataset = dataset.PJDataset(mode='val', stage=s)
    test_dataset = dataset.PJDataset(mode='test', stage=s)

    cfg = util.default_cfg()
    cfg['train'] = train_dataset
    cfg['val'] = val_dataset
    cfg['test'] = test_dataset
    cfg['batch'] = 64
    cfg['lr'] = 0.000001
    cfg['model'] = 'resnet_pj'
    cfg['model_dir'] = 'modeldir/stage%d/resnet_pj' % s
    cfg['collate'] = default_collate
    cfg['instance'] = _train_si

    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    model = nn.DataParallel(SiNet().cuda())
    if os.path.exists(model_pth):
        print("load pretrained model", model_pth)
        model.load_state_dict(torch.load(model_pth))

    run_train(model, cfg)


def train_naggn(s=2):
    train_dataset = dataset.DrosophilaDataset(mode='train', stage=s)
    val_dataset = dataset.DrosophilaDataset(mode='val', stage=s)
    test_dataset = dataset.DrosophilaDataset(mode='test', stage=s)

    cfg = util.default_cfg()
    cfg['train'] = train_dataset
    cfg['val'] = val_dataset
    cfg['test'] = test_dataset
    cfg['batch'] = 64
    cfg['lr'] = 0.00001
    cfg['model'] = 'naggn'
    cfg['model_dir'] = 'modeldir/stage%d/naggn' % s
    cfg['collate'] = dataset.fly_collate_fn
    cfg['instance'] = _train_mi

    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    model = nn.DataParallel(naggn.NAggN().cuda())
    if os.path.exists(model_pth):
        print("load pretrained model", model_pth)
        model.load_state_dict(torch.load(model_pth))

    run_train(model, cfg)


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
    cfg['instance'] = _train_si

    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    model = nn.DataParallel(dragn.DRAGN().cuda())
    if os.path.exists(model_pth):
        print("load pretrained model", model_pth)
        model.load_state_dict(torch.load(model_pth))
    run_train(model, cfg)


def _train_si(model, sample_batched):
    sgene, img, label = sample_batched
    # inputs = torch.from_numpy(img).type(torch.cuda.FloatTensor)
    # gt = torch.from_numpy(label).type(torch.cuda.FloatTensor)
    inputs = img.type(torch.cuda.FloatTensor)
    gt = label.type(torch.cuda.FloatTensor)
    model.zero_grad()
    predict = model(inputs)
    return sgene, predict, gt


def _train_mi(model, sample_batched):
    sgene, img, label, nslice = sample_batched
    inputs = torch.from_numpy(img).type(torch.cuda.FloatTensor)
    gt = torch.from_numpy(label).type(torch.cuda.FloatTensor)
    nslice = torch.from_numpy(nslice)
    model.zero_grad()
    predict = model(inputs, nslice)
    return sgene, predict, gt


def run_train(model, cfg):
    train_loader = DataLoader(cfg['train'], batch_size=cfg['batch'],
                              shuffle=True, num_workers=cfg['nworker'],
                              collate_fn=cfg['collate'])
    model_pth = os.path.join(cfg['model_dir'], "model.pth")
    writer = tensorboardX.SummaryWriter(cfg['model_dir'])
    cfg['writer'] = writer

    criterion = cfg['criterion']
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['decay'])
    if cfg['scheduler']:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'max',
            factor=cfg['factor'], patience=cfg['patience'])

    step = 0
    for e in range(cfg['epochs']):
        print("----run train---", cfg['model'], e)
        model.train()
        st = time.time()

        cfg['step'] = e
        for i_batch, sample_batched in enumerate(train_loader):
            sgene, predict, gt = cfg['instance'](model, sample_batched)
            loss = criterion(predict, gt)
            loss.backward()
            optimizer.step()

            writer.add_scalar("loss", loss, step)
            step += 1

        et = time.time()
        writer.add_scalar("train time", et - st, e)

        val_loss, lab_f1_macro = run_val(model, cfg)
        print("val loss:", val_loss, "\tf1:", lab_f1_macro)
        if cfg['scheduler']:
            scheduler.step(lab_f1_macro)
            for g in optimizer.param_groups:
                writer.add_scalar("lr", g['lr'], e)

        if e == 0:
            start_loss = val_loss
            min_loss = start_loss
            max_f1 = 0.0

        # if val_loss > 2 * min_loss:
        #     print("early stopping at %d" % e)
        #     break
        # run_test(model, cfg)

        if min_loss > val_loss or lab_f1_macro > max_f1:
            if min_loss > val_loss:
                min_loss = val_loss
                print("----save best epoch:%d, loss:%f---" % (e, val_loss))
            if lab_f1_macro > max_f1:
                max_f1 = lab_f1_macro
                print("----save best epoch:%d, f1:%f---" % (e, max_f1))
            torch.save(model.state_dict(), model_pth)
            run_test(model, cfg)


def run_val(model, cfg):
    print("----run val---", cfg['model'], cfg['step'])
    model.eval()
    with torch.no_grad():
        val_loader = DataLoader(cfg['val'], batch_size=cfg['batch'],
                                shuffle=False, num_workers=cfg['nworker'],
                                collate_fn=cfg['collate'])
        criterion = cfg['criterion']
        np_label = []
        np_pd = []
        np_score = []
        tot_loss = 0.0

        for i_batch, sample_batched in enumerate(val_loader):
            sgene, predict, gt = cfg['instance'](model, sample_batched)
            loss = criterion(predict, gt)
            tot_loss += loss

            val_pd = util.threshold_tensor_batch(predict)
            np_pd.append(val_pd.data.cpu().numpy())
            np_score.append(predict.data.cpu().numpy())
            np_label.append(gt.data.cpu().numpy())

        np_label = np.concatenate(np_label)
        np_pd = np.concatenate(np_pd)
        np_score = np.concatenate(np_score)

        tot_loss = tot_loss / len(val_loader)
        cfg['writer'].add_scalar("val loss", tot_loss.item(), cfg['step'])
        lab_f1_macro = util.torch_metrics(np_label, np_pd, cfg['writer'],
                                          cfg['step'], score=np_score)

        return tot_loss.item(), lab_f1_macro


def run_test(model, cfg):
    print("----run test---", cfg['model'], cfg['step'])
    model.eval()
    with torch.no_grad():
        test_loader = DataLoader(cfg['test'], batch_size=cfg['batch'],
                                 shuffle=False, num_workers=cfg['nworker'],
                                 collate_fn=cfg['collate'])
        np_label = []
        np_pd = []
        np_score = []
        np_sgene = []
        for i_batch, sample_batched in enumerate(test_loader):
            sgene, predict, gt = cfg['instance'](model, sample_batched)
            test_pd = util.threshold_tensor_batch(predict)
            np_pd.append(test_pd.data.cpu().numpy())
            np_sgene.extend(sgene)
            np_label.append(gt.data.cpu().numpy())
            np_score.append(predict.data.cpu().numpy())

        np_label = np.concatenate(np_label)
        np_pd = np.concatenate(np_pd)
        np_score = np.concatenate(np_score)
        # util.torch_metrics(np_label, np_pd, cfg['writer'],
        #                    cfg['step'], mode='test', score=np_score)
        pth = os.path.join(cfg['model_dir'], 'metrics.csv')
        util.write_metrics(pth, np_label, np_pd, np_score)

        np_target = [' '.join([str(x) for x in np.where(item)[0]])
                     for item in np_pd]

        df = pd.DataFrame({'Gene': np_sgene, 'Predicted': np_target})

        result = os.path.join(cfg['model_dir'], "%s_%d.csv"
                              % (cfg['model'], cfg['step']))
        df.to_csv(result, header=True, sep=',', index=False)


if __name__ == "__main__":
    # train_naggn(s=3)
    # train_resnet_si(s=6)
    train_resnet_pj(s=6)
