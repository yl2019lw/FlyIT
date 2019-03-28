#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset
from collections import Counter


def load_all_data():
    img = pd.read_csv('data/standard_images.csv')
    ann = pd.read_csv('data/standard_annotations.csv')

    stages = [2, 3, 4, 5, 6]
    directions = ['lateral', 'dorsal', 'ventral']

    img = pd.read_csv('data/standard_images.csv')
    ann = pd.read_csv('data/standard_annotations.csv')

    img = img[img.stage.isin(stages)]
    img = img[img.direction.isin(directions)]

    ann = ann[ann.stage.isin(stages)]

    i_gene = img['gene'].unique()
    a_gene = ann['gene'].unique()

    gene = np.intersect1d(i_gene, a_gene)
    imgdir = os.path.join('data/pic')

    d = {}
    for g in gene:
        for s in stages:
            sid = '%s_s%s' % (g, s)

            sid_pdimg = img.loc[img.gene == g].loc[img.stage == s]
            sid_imgs = []
            if sid_pdimg.empty:
                continue
            else:
                for item in sid_pdimg['image_url'].values:
                    sid_imgs += [x for x in list(eval(item))
                                 if os.path.exists(os.path.join(imgdir, x))]

            sid_pdann = ann.loc[ann.gene == g].loc[ann.stage == s]
            sid_anns = []
            if sid_pdann.empty:
                continue
            else:
                for item in sid_pdann['annotation'].values:
                    sid_anns += list(eval(item))

            sid_imgs = list(set(sid_imgs))
            sid_anns = list(set(sid_anns))

            if len(sid_imgs) and len(sid_anns):
                d[sid] = {}
                d[sid]['img'] = sid_imgs
                d[sid]['ann'] = sid_anns

    return d


def filter_top_cv(d, k=10):
    '''return samples have top k cv'''
    all_cv = []

    all_sids = sorted(d.keys())

    for sid in all_sids:
        all_cv += d[sid]['ann']

    count = Counter(all_cv)
    top_cv = [x[0] for x in count.most_common(k)]

    filter_d = {}

    for sid in all_sids:
        for ann in d[sid]['ann']:
            if ann not in top_cv:
                continue

            if sid not in filter_d:
                filter_d[sid] = {}
                filter_d[sid]['ann'] = []
            filter_d[sid]['ann'].append(ann)

        if sid in filter_d:
            filter_d[sid]['img'] = d[sid]['img']

    if len(top_cv) < k:
        print("top cv less than k", count)

    return filter_d, top_cv


def stat(k=10):
    d = load_all_data()
    filter_d, top_cv = filter_top_cv(d, k)
    print("---top cv---", top_cv)

    sids = sorted(filter_d.keys())
    imgs = []
    all_cv = []
    for sid in sids:
        imgs.extend(d[sid]['img'])
        all_cv += d[sid]['ann']

    print("sids", len(sids), "imgs", len(imgs))


def all_k_stat():
    for k in [10, 20, 30]:
        print("------all k stat---------", k)
        d = load_all_data()
        filter_d, top_cv = filter_top_cv(d, k)

        sids = sorted(filter_d.keys())
        np.random.seed(286501567)
        np.random.shuffle(sids)
        ts = int(len(sids) * 0.5)
        vs = int(len(sids) * 0.4)

        train_sids = sids[:vs]
        val_sids = sids[vs:ts]
        test_sids = sids[ts:]
        train_imgs = []
        val_imgs = []
        test_imgs = []

        for sid in train_sids:
            train_imgs.extend(d[sid]['img'])

        for sid in val_sids:
            val_imgs.extend(d[sid]['img'])

        for sid in test_sids:
            test_imgs.extend(d[sid]['img'])

        print("train sids:", len(train_sids), "train imgs:", len(train_imgs))
        print("val sids:", len(val_sids), "train imgs:", len(val_imgs))
        print("test sids:", len(test_sids), "train imgs:", len(test_imgs))
        print("\n")


def split_train_val_test(sids, labels):
    np.random.seed(286501567)
    np.random.shuffle(sids)
    ts = int(len(sids) * 0.5)
    vs = int(len(sids) * 0.4)

    return sids[:vs], sids[vs:ts], sids[ts:]


def stratify_split_train_val_test(sids, labels):
    '''skmultilearn needs X as nsampel x ndim inputs'''
    np.random.seed(286501567)
    inputs = np.expand_dims(sids, axis=-1)
    from skmultilearn.model_selection import iterative_train_test_split
    X, y, X_test, y_test = iterative_train_test_split(
        inputs, labels, test_size=0.5)
    X_train, y_train, X_val, y_val = iterative_train_test_split(
        X, y, test_size=0.2)
    return X_train.squeeze(), X_val.squeeze(), X_test.squeeze()


def generate_pj_samples(d, sids, shuffle=False, count=4):
    sid_pj_imgs = []
    for sid in sids:
        all_sid_imgs = d[sid]['img']
        if shuffle:
            np.random.shuffle(all_sid_imgs)
        chunk_imgs = [all_sid_imgs[i:i+count]
                      for i in range(0, len(all_sid_imgs), count)]
        sid_pj_imgs.extend([(sid, c) for c in chunk_imgs])
    return sid_pj_imgs


class SIDataset(Dataset):

    def __init__(self, mode='train', k=10):
        super(SIDataset, self).__init__()
        self.mode = mode
        self.db, self.top_cv = filter_top_cv(load_all_data(), k)
        self.nclass = len(self.top_cv)
        sids = np.array(list(sorted(self.db.keys())))

        labels = [self._get_sid_label(sid) for sid in sids]
        labels = np.stack(labels, axis=0)

        train_sids, val_sids, test_sids = split_train_val_test(sids, labels)

        if mode == 'train':
            self.sids = train_sids
        elif mode == 'val':
            self.sids = val_sids
        else:
            self.sids = test_sids

        self.sid_imgs = [(sid, img) for sid in self.sids
                         for img in self.db[sid]['img']]

    def _get_sid_label(self, sid):
        sid_anns = self.db[sid]['ann']
        anns = np.zeros(self.nclass)
        for ann in sid_anns:
            anns[self.top_cv.index(ann)] = 1
        return anns

    def __len__(self):
        return len(self.sid_imgs)

    def __getitem__(self, idx):
        sid, img = self.sid_imgs[idx]
        imgpth = os.path.join('data/pic', img)
        nimg = cv2.imread(imgpth, -1)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        nimg = nimg.transpose(2, 0, 1)

        anns = self._get_sid_label(sid)

        # mean = np.mean(nimg, axis=(1, 2), keepdims=True)
        # std = np.std(nimg, axis=(1, 2), keepdims=True)
        # nimg = (nimg - mean) / std

        return sid, nimg, anns


class PJDataset(Dataset):

    def __init__(self, mode='train', k=10):
        super(PJDataset, self).__init__()
        self.mode = mode
        self.db, self.top_cv = filter_top_cv(load_all_data(), k)
        self.nclass = len(self.top_cv)
        sids = np.array(list(sorted(self.db.keys())))

        labels = [self._get_sid_label(sid) for sid in sids]
        labels = np.stack(labels, axis=0)

        train_sids, val_sids, test_sids = split_train_val_test(sids, labels)

        if mode == 'train':
            self.sids = train_sids
            self.sid_imgs = generate_pj_samples(self.db, self.sids,
                                                shuffle=True)

        elif mode == 'val':
            self.sids = val_sids
            self.sid_imgs = generate_pj_samples(self.db, self.sids)

        else:
            self.sids = test_sids
            self.sid_imgs = generate_pj_samples(self.db, self.sids)

    def _get_sid_label(self, sid):
        sid_anns = self.db[sid]['ann']
        anns = np.zeros(self.nclass)
        for ann in sid_anns:
            anns[self.top_cv.index(ann)] = 1
        return anns

    def __len__(self):
        return len(self.sid_imgs)

    def __getitem__(self, idx):
        sid, imgs = self.sid_imgs[idx]

        raw_nimgs = []
        for img in imgs:
            imgpth = os.path.join('data/pic', img)
            nimg = cv2.imread(imgpth, -1)
            nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
            nimg = nimg.transpose(2, 0, 1)
            raw_nimgs.append(nimg)
        raw_nimgs = np.stack(raw_nimgs, axis=0)

        n, c, h, w = raw_nimgs.shape
        pj_nimg = np.zeros((c, 2*h, 2*w), dtype=np.float)
        if n == 4:
            pj_nimg[:, 0:h, 0:w] = raw_nimgs[0]
            pj_nimg[:, 0:h, w:2*w] = raw_nimgs[1]
            pj_nimg[:, h:2*h, 0:w] = raw_nimgs[2]
            pj_nimg[:, h:2*h, w:2*w] = raw_nimgs[3]
        elif n == 3:
            pj_nimg[:, 0:h, 0:w] = raw_nimgs[0]
            pj_nimg[:, 0:h, w:2*w] = raw_nimgs[1]
            pj_nimg[:, h:2*h, 0:w] = raw_nimgs[2]
            pj_nimg[:, h:2*h, w:2*w] = raw_nimgs[0]
        elif n == 2:
            pj_nimg[:, 0:h, 0:w] = raw_nimgs[0]
            pj_nimg[:, 0:h, w:2*w] = raw_nimgs[1]
            pj_nimg[:, h:2*h, 0:w] = raw_nimgs[0]
            pj_nimg[:, h:2*h, w:2*w] = raw_nimgs[1]
        elif n == 1:
            pj_nimg[:, 0:h, 0:w] = raw_nimgs[0]
            pj_nimg[:, 0:h, w:2*w] = raw_nimgs[0]
            pj_nimg[:, h:2*h, 0:w] = raw_nimgs[0]
            pj_nimg[:, h:2*h, w:2*w] = raw_nimgs[0]
        else:
            raise Exception("invalid img numbers in PJDataset:", n)

        anns = self._get_sid_label(sid)

        return sid, pj_nimg, anns


if __name__ == "__main__":
    # stat(k=30)
    all_k_stat()
