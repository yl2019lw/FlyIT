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
from tqdm import tqdm
import dataset
import naggn
import dragn
import transformer
import util
import sinet


def train_senet_si(s=2, k=10, val_index=4):
    train_dataset = dataset.SIDataset(
        mode='train', stage=s, k=k, val_index=val_index)
    val_dataset = dataset.SIDataset(
        mode='val', stage=s, k=k, val_index=val_index)
    test_dataset = dataset.SIDataset(
        mode='test', stage=s, k=k, val_index=val_index)

    cfg = util.default_cfg()
    cfg['train'] = train_dataset
    cfg['val'] = val_dataset
    cfg['test'] = test_dataset
    cfg['batch'] = 128
    cfg['scheduler'] = True
    cfg['decay'] = 0.01
    cfg['lr'] = 0.0001
    cfg['model'] = 'senet_si_k%d_val%d' % (k, val_index)
    cfg['model_dir'] = 'modeldir/stage%d/senet_si_k%d_val%d' % (
        s, k, val_index)
    cfg['collate'] = default_collate
    cfg['instance'] = _train_si

    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    model = nn.DataParallel(sinet.FlySENet(k=k).cuda())
    if os.path.exists(model_pth):
        ckp = torch.load(model_pth)
        model.load_state_dict(ckp['model'])
        cfg['step'] = ckp['epoch'] + 1
        print("load pretrained model", model_pth, "start epoch:", cfg['step'])

    run_train(model, cfg)


def train_resnet_si(s=2, k=10, val_index=4):
    train_dataset = dataset.SIDataset(
        mode='train', stage=s, k=k, val_index=val_index)
    val_dataset = dataset.SIDataset(
        mode='val', stage=s, k=k, val_index=val_index)
    test_dataset = dataset.SIDataset(
        mode='test', stage=s, k=k, val_index=val_index)

    cfg = util.default_cfg()
    cfg['train'] = train_dataset
    cfg['val'] = val_dataset
    cfg['test'] = test_dataset
    cfg['batch'] = 64
    # from loss import FECLoss
    # cfg['criterion'] = FECLoss(alpha=64)
    cfg['epochs'] = 1000
    cfg['scheduler'] = True
    cfg['decay'] = 0.01
    cfg['lr'] = 0.0001
    cfg['patience'] = 20
    # cfg['model'] = 'resnet_si_k%d_val%d' % (k, val_index)
    # cfg['model_dir'] = 'modeldir/stage%d/resnet_si_k%d_val%d' % (
    #     s, k, val_index)
    cfg['model'] = 'smallnet_si_k%d_val%d' % (k, val_index)
    cfg['model_dir'] = 'modeldir/stage%d/smallnet_si_k%d_val%d' % (
        s, k, val_index)
    cfg['collate'] = default_collate
    cfg['instance'] = _train_si

    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    # model = nn.DataParallel(SiNet(nblock=4, k=k).cuda())
    model = nn.DataParallel(sinet.SmallNet(k=k).cuda())
    if os.path.exists(model_pth):
        # print("load pretrained model", model_pth)
        # model.load_state_dict(torch.load(model_pth))
        ckp = torch.load(model_pth)
        model.load_state_dict(ckp['model'])
        cfg['step'] = ckp['epoch'] + 1
        print("load pretrained model", model_pth, "start epoch:", cfg['step'])

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
    model = nn.DataParallel(sinet.SiNet(nblock=4).cuda())
    if os.path.exists(model_pth):
        ckp = torch.load(model_pth)
        model.load_state_dict(ckp['model'])
        cfg['step'] = ckp['epoch'] + 1
        print("load pretrained model", model_pth, "start epoch:", cfg['step'])

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
    cfg['lr'] = 0.000001
    cfg['model'] = 'naggn_l1'
    cfg['model_dir'] = 'modeldir/stage%d/naggn_l1' % s
    cfg['collate'] = dataset.fly_collate_fn
    cfg['instance'] = _train_mi

    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    model = nn.DataParallel(naggn.NAggN(agg='l1').cuda())
    if os.path.exists(model_pth):
        ckp = torch.load(model_pth)
        model.load_state_dict(ckp['model'])
        cfg['step'] = ckp['epoch'] + 1
        print("load pretrained model", model_pth, "start epoch:", cfg['step'])

    run_train(model, cfg)


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
    cfg['instance'] = _train_mi

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
    cfg['instance'] = _train_mi

    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    model = nn.DataParallel(transformer.E2ETransformer().cuda())
    if os.path.exists(model_pth):
        ckp = torch.load(model_pth)
        model.load_state_dict(ckp['model'])
        cfg['step'] = ckp['epoch'] + 1
        print("load pretrained model", model_pth, "start epoch:", cfg['step'])

    run_train(model, cfg)


def _train_si(model, sample_batched):
    sgene, img, label = sample_batched
    inputs = img.type(torch.cuda.FloatTensor)
    gt = label.type(torch.cuda.FloatTensor)
    model.zero_grad()
    predict = model(inputs)
    return sgene, predict, gt


def _train_mi(model, sample_batched):
    sgene, img, label, nslice = sample_batched
    inputs = img.type(torch.cuda.FloatTensor)
    gt = label.type(torch.cuda.FloatTensor)
    nslice = nslice.type(torch.cuda.IntTensor)
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

    step = cfg['step'] * len(train_loader)
    min_loss = 1e8
    max_f1 = 0.0
    for e in range(cfg['step'], cfg['epochs']):
        print("----run train---", cfg['model'], e)
        model.train()
        st = time.time()

        cfg['step'] = e
        np_label = []
        np_pd = []
        np_score = []
        for i_batch, sample_batched in tqdm(
                enumerate(train_loader), total=len(train_loader)):
            sgene, predict, gt = cfg['instance'](model, sample_batched)
            loss = criterion(predict, gt)
            loss.backward()
            optimizer.step()

            writer.add_scalar("loss", loss, step)
            step += 1

            val_pd = util.threshold_tensor_batch(predict)
            np_pd.append(val_pd.data.cpu().numpy())
            np_score.append(predict.data.cpu().numpy())
            np_label.append(gt.data.cpu().numpy())

        np_label = np.concatenate(np_label)
        np_pd = np.concatenate(np_pd)
        np_score = np.concatenate(np_score)

        et = time.time()
        writer.add_scalar("train time", et - st, e)

        util.torch_metrics(np_label, np_pd, cfg['writer'],
                           cfg['step'], mode='train', score=np_score)

        val_loss, lab_f1_macro = run_val(model, cfg)
        print("val loss:", val_loss, "\tf1:", lab_f1_macro)
        for g in optimizer.param_groups:
            writer.add_scalar("lr", g['lr'], e)

            if cfg['scheduler'] and g['lr'] > 1e-5:
                scheduler.step(lab_f1_macro)
                break

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
            # torch.save(model.state_dict(), model_pth)
            torch.save({'epoch': e, 'model': model.state_dict()}, model_pth)
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


def sequence_train():
    import multiprocessing as mp
    for s in [6, 5, 4, 3, 2]:
        for val_index in [4, 3, 2, 1, 0]:
            p = mp.Process(target=train_resnet_si, args=(s, 10, val_index))
            p.start()
            p.join()


def run_test_score(model, cfg):
    print("----run test score---", cfg['model'])
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

        return np_score, np_label


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
            cfg['instance'] = _train_si
            cfg['model'] = m_dir

            np_score, np_label = run_test_score(model, cfg)
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


if __name__ == "__main__":
    # train_transformer(s=2)
    # train_naggn(s=2)
    # train_prefv_naggn(s=6)
    train_resnet_si(s=2, k=10, val_index=4)
    # train_resnet_pj(s=2)
    # sequence_train()
    # run_kfold_test()
    # train_senet_si(s=2, k=10, val_index=4)
