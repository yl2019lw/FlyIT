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
from torch.utils.data import Subset
from tqdm import tqdm
import dataset
import util
import sinet


def train_smallnet_si_boost(s=2, k=10):
    cfg = util.default_cfg()

    train_dataset = dataset.StratifySIDataset(
        mode='train', stage=s, k=k)
    val_dataset = dataset.StratifySIDataset(
        mode='val', stage=s, k=k)
    test_dataset = dataset.StratifySIDataset(
        mode='test', stage=s, k=k)

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
    cfg['instance'] = _train_si
    cfg['criterion'] = torch.nn.BCELoss(reduction='none')

    cfg['model'] = 'smallnet_si_k%d_boost' % (k)
    cfg['model_dir'] = 'modeldir/stage%d/smallnet_si_k%d_boost' % (s, k)
    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    model = nn.DataParallel(sinet.SmallNet(k=k).cuda())
    if os.path.exists(model_pth):
        ckp = torch.load(model_pth)
        model.load_state_dict(ckp['model'])
        cfg['step'] = ckp['epoch'] + 1
        print("load pretrained model", model_pth, "start epoch:", cfg['step'])

    run_train(model, cfg)


def train_smallnet_si_boost_fec(s=2, k=10):
    cfg = util.default_cfg()

    train_dataset = dataset.StratifySIDataset(
        mode='train', stage=s, k=k)
    val_dataset = dataset.StratifySIDataset(
        mode='val', stage=s, k=k)
    test_dataset = dataset.StratifySIDataset(
        mode='test', stage=s, k=k)

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
    cfg['instance'] = _train_si

    from loss import FECLoss
    cfg['criterion'] = FECLoss(alpha=64, reduction='none')

    cfg['model'] = 'smallnet_si_k%d_boost_fec1' % (k)
    cfg['model_dir'] = 'modeldir/stage%d/smallnet_si_k%d_boost_fec1' % (s, k)
    model_pth = os.path.join(cfg['model_dir'], 'model.pth')
    model = nn.DataParallel(sinet.SmallNet(k=k).cuda())
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


def run_train(model, cfg):
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

    nsample = len(cfg['train'])
    sample_weights = np.ones(nsample, dtype='float')
    # origin bce loss
    sample_loss = np.zeros(nsample, dtype='float')

    step = cfg['step'] * len(cfg['train'])
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

        shuffle_idx = np.array(range(len(cfg['train'])))
        np.random.shuffle(shuffle_idx)
        train_dataset = Subset(cfg['train'], shuffle_idx)

        train_loader = DataLoader(train_dataset, batch_size=cfg['batch'],
                                  shuffle=False, num_workers=cfg['nworker'],
                                  collate_fn=cfg['collate'])
        for i_batch, sample_batched in tqdm(
                enumerate(train_loader), total=len(train_loader)):
            sgene, predict, gt = cfg['instance'](model, sample_batched)

            st_idx = i_batch * cfg['batch']
            end_idx = st_idx + len(sgene)
            batch_idx = shuffle_idx[st_idx:end_idx]

            # orig_loss = F.binary_cross_entropy(predict, gt, reduction='none')
            orig_loss = criterion(predict, gt)
            orig_loss = torch.sum(orig_loss, dim=1) / predict.shape[-1]
            weight_loss = torch.from_numpy(
                sample_weights[batch_idx]).type(
                torch.cuda.FloatTensor) * orig_loss
            loss = torch.mean(weight_loss)

            writer.add_scalar("loss_weight", loss, step)
            writer.add_scalar("loss_orig", torch.mean(orig_loss), step)

            # loss = criterion(predict, gt)
            loss.backward()
            optimizer.step()

            # writer.add_scalar("loss", loss, step)
            step += 1

            val_pd = util.threshold_tensor_batch(predict)
            np_pd.append(val_pd.data.cpu().numpy())
            np_score.append(predict.data.cpu().numpy())
            np_label.append(gt.data.cpu().numpy())

            sample_loss[batch_idx] = orig_loss.clone().cpu().data.numpy()

        exp_loss = np.exp(sample_loss)
        exp_tot = np.sum(exp_loss)
        sample_weights = nsample * (exp_loss / exp_tot)
        sample_loss = np.zeros(nsample)

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
            tot_loss += torch.mean(loss)

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
    # train_smallnet_si_boost(s=2, k=10)
    train_smallnet_si_boost_fec(s=2, k=10)
