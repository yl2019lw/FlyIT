#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter
from imgaug import augmenters as iaa


def load_by_stage(stage=1):
    img = pd.read_csv('data/standard_images.csv')
    ann = pd.read_csv('data/standard_annotations.csv')

    if stage == -1:
        # use all stage data
        img = img[img['stage'] != 1]
        ann = ann[ann['stage'] != 1]
    else:
        img = img[img['stage'] == stage]
        ann = ann[ann['stage'] == stage]

    i_gene = img['gene'].unique()
    a_gene = ann['gene'].unique()

    gene = np.intersect1d(i_gene, a_gene)
    imgdir = os.path.join('data/pic')

    d = {}
    for g in gene:
        d[g] = {}
        gene_imgs = []
        for item in img[img['gene'] == g]['image_url'].values:
            gene_imgs += [x for x in list(eval(item))
                          if os.path.exists(os.path.join(imgdir, x))]

        gene_annotations = []
        for item in ann[ann['gene'] == g]['annotation'].values:
            gene_annotations += list(eval(item))

        d[g]['img'] = list(set(gene_imgs))
        d[g]['ann'] = list(set(gene_annotations))

    return d


def filter_top_cv(d, k=10):
    '''return genes have top k cv'''
    all_cv = []
    for g in d.keys():
        all_cv += d[g]['ann']
    count = Counter(all_cv)
    top_cv = [x[0] for x in count.most_common(k)]
    fd = {}
    for g in d.keys():
        for ann in d[g]['ann']:
            if ann not in top_cv:
                continue
            if g not in fd:
                fd[g] = {}
                fd[g]['ann'] = []
            fd[g]['ann'].append(ann)

        if g in fd:
            fd[g]['img'] = d[g]['img']
    if len(top_cv) < k:
        print("top cv less than k", count)
    return fd, top_cv


def generate_pj_samples(d, genes, shuffle=False, count=4):
    gene_imgs = []
    for gene in genes:
        ag_imgs = d[gene]['img']
        if shuffle:
            np.random.shuffle(ag_imgs)
        chunk_imgs = [ag_imgs[i:i+count]
                      for i in range(0, len(ag_imgs), count)]
        gene_imgs.extend([(gene, c) for c in chunk_imgs])
    return gene_imgs


def aug():
    hf = iaa.Fliplr(0.5)
    vf = iaa.Flipud(0.5)
    contrast = iaa.Sometimes(
        0.5, iaa.ContrastNormalization((0.8, 1.2)))
    # color = iaa.Sometimes(
    #     0.5, iaa.AddToHueAndSaturation((-10, 10), per_channel=True))
    blur = iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.1)))
    trfm = iaa.Sequential([hf, vf, contrast, blur])
    return trfm


def stratify_split(genes, labels):
    '''skmultilearn needs X as nsampel x ndim inputs'''
    np.random.seed(286501567)
    inputs = np.expand_dims(genes, axis=-1)
    from skmultilearn.model_selection import iterative_train_test_split
    X, y, X_test, y_test = iterative_train_test_split(
        inputs, labels, test_size=0.5)
    X_train, y_train, X_val, y_val = iterative_train_test_split(
        X, y, test_size=0.2)
    return X_train.squeeze(), X_val.squeeze(), X_test.squeeze()


class StratifySIDataset(Dataset):

    def __init__(self, mode='train', stage=2, k=10):
        super(StratifySIDataset, self).__init__()
        self.mode = mode
        self.db, self.top_cv = filter_top_cv(load_by_stage(stage), k)
        self.nclass = len(self.top_cv)
        genes = np.array(list(sorted(self.db.keys())))

        labels = [self._get_gene_label(g) for g in genes]
        labels = np.stack(labels, axis=0)

        train_genes, val_genes, test_genes = stratify_split(genes, labels)

        if mode == 'train':
            self.genes = train_genes
            self.aug = aug()
        elif mode == 'val':
            self.genes = val_genes
        else:
            self.genes = test_genes

        self.gene_imgs = [(gene, img) for gene in self.genes
                          for img in self.db[gene]['img']]

    def _get_gene_label(self, gene):
        gene_anns = self.db[gene]['ann']
        anns = np.zeros(self.nclass)
        for ann in gene_anns:
            anns[self.top_cv.index(ann)] = 1
        return anns

    def __len__(self):
        return len(self.gene_imgs)

    def __getitem__(self, idx):
        gene, img = self.gene_imgs[idx]
        imgpth = os.path.join('data/pic', img)
        nimg = cv2.imread(imgpth, -1)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        nimg = nimg.transpose(2, 0, 1)

        anns = self._get_gene_label(gene)

        if self.mode == 'train':
            nimg = self.aug.augment_image(nimg)

        mean = np.mean(nimg, axis=(1, 2), keepdims=True)
        std = np.std(nimg, axis=(1, 2), keepdims=True)
        nimg = (nimg - mean) / std

        return gene, nimg, anns


class StratifyPJDataset(Dataset):

    def __init__(self, mode='train', stage=2, k=10):
        super(StratifyPJDataset, self).__init__()
        self.mode = mode
        self.db, self.top_cv = filter_top_cv(load_by_stage(stage), k)
        self.nclass = len(self.top_cv)
        genes = np.array(list(sorted(self.db.keys())))

        labels = [self._get_gene_label(g) for g in genes]
        labels = np.stack(labels, axis=0)

        train_genes, val_genes, test_genes = stratify_split(genes, labels)

        if mode == 'train':
            self.genes = train_genes
            self.aug = aug()
            self.gene_imgs = generate_pj_samples(
                self.db, self.genes, shuffle=True)
        elif mode == 'val':
            self.genes = val_genes
            self.gene_imgs = generate_pj_samples(self.db, self.genes)
        else:
            self.genes = test_genes
            self.gene_imgs = generate_pj_samples(self.db, self.genes)

    def _get_gene_label(self, gene):
        gene_anns = self.db[gene]['ann']
        anns = np.zeros(self.nclass)
        for ann in gene_anns:
            anns[self.top_cv.index(ann)] = 1
        return anns

    def __len__(self):
        return len(self.gene_imgs)

    def __getitem__(self, idx):
        gene, imgs = self.gene_imgs[idx]

        raw_nimgs = []
        for img in imgs:
            imgpth = os.path.join('data/pic', img)
            nimg = cv2.imread(imgpth, -1)
            nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
            nimg = nimg.transpose(2, 0, 1)
            if self.mode == 'train':
                nimg = self.aug.augment_image(nimg)
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

        anns = self._get_gene_label(gene)

        return gene, pj_nimg, anns


def split_train_val(l, val_index=4, kfold=5):
    '''kfold split train val'''
    csize = len(l) // kfold
    chunk_item = [l[c:c+csize] for c in range(0, len(l), csize)]

    train_item = []
    val_item = []

    all_index = list(range(kfold))
    all_index.remove(val_index)

    val_item = chunk_item[val_index]
    for i in all_index:
        train_item.extend(chunk_item[i])

    return train_item, val_item


class SIDataset(Dataset):
    '''Single Instance Dataset, k class'''

    def __init__(self, mode='train', stage=2, k=10, val_index=4):
        super(SIDataset, self).__init__()
        self.mode = mode
        self.db, self.top_cv = filter_top_cv(load_by_stage(stage), k)
        self.nclass = len(self.top_cv)
        genes = list(sorted(self.db.keys()))
        half = int(len(genes) * 0.5)
        tv_genes = genes[:half]
        test_genes = genes[half:]
        train_genes, val_genes = split_train_val(tv_genes, val_index)
        if mode == 'train':
            self.genes = train_genes
            self.aug = aug()
        elif mode == 'val':
            self.genes = val_genes
        else:
            self.genes = test_genes

        self.gene_imgs = [(gene, img) for gene in self.genes
                          for img in self.db[gene]['img']]

    def __len__(self):
        return len(self.gene_imgs)

    def __getitem__(self, idx):
        gene, img = self.gene_imgs[idx]
        imgpth = os.path.join('data/pic', img)
        nimg = cv2.imread(imgpth, -1)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        nimg = nimg.transpose(2, 0, 1)

        gene_anns = self.db[gene]['ann']
        anns = np.zeros(self.nclass)
        for ann in gene_anns:
            anns[self.top_cv.index(ann)] = 1

        if self.mode == 'train':
            nimg = self.aug.augment_image(nimg)

        return gene, nimg, anns


class PJDataset(Dataset):
    '''Pin Jie Dataset as ltg does'''

    def __init__(self, mode='train', stage=2, k=10):
        super(PJDataset, self).__init__()
        self.db, self.top_cv = filter_top_cv(load_by_stage(stage), k)
        self.nclass = len(self.top_cv)
        genes = list(sorted(self.db.keys()))
        tl = int(len(genes) * 0.4)
        vl = int(len(genes) * 0.5)
        if mode == 'train':
            self.genes = genes[:tl]
        elif mode == 'test':
            self.genes = genes[vl:]
        else:
            self.genes = genes[tl:vl]

        self.gene_imgs = generate_pj_samples(self.db, self.genes)

    def __len__(self):
        return len(self.gene_imgs)

    def __getitem__(self, idx):
        gene, imgs = self.gene_imgs[idx]

        raw_nimgs = []
        for img in imgs:
            imgpth = os.path.join('data/pic', img)
            nimg = cv2.imread(imgpth, -1)
            nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
            raw_nimgs.append(nimg.transpose(2, 0, 1))
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

        gene_anns = self.db[gene]['ann']
        anns = np.zeros(self.nclass)
        for ann in gene_anns:
            anns[self.top_cv.index(ann)] = 1

        return gene, pj_nimg, anns


class DrosophilaDataset(Dataset):

    def __init__(self, mode='train', stage=2, k=10):
        self.db, self.top_cv = filter_top_cv(load_by_stage(stage), k)
        self.nclass = len(self.top_cv)
        genes = list(sorted(self.db.keys()))
        tl = int(len(genes) * 0.4)
        vl = int(len(genes) * 0.5)
        if mode == 'train':
            self.genes = genes[:tl]
        elif mode == 'test':
            self.genes = genes[vl:]
        else:
            self.genes = genes[tl:vl]

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, idx):
        gene = self.genes[idx]

        gene_imgs = self.db[gene]['img']
        imgs = []
        for img in gene_imgs:
            imgpth = os.path.join('data/pic', img)
            nimg = cv2.imread(imgpth, -1)
            nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
            imgs.append(nimg.transpose(2, 0, 1))
        imgs = np.stack(imgs).astype(np.float)

        gene_anns = self.db[gene]['ann']
        anns = np.zeros(self.nclass)
        for ann in gene_anns:
            anns[self.top_cv.index(ann)] = 1

        # print("dataset gene", gene, imgs.shape, anns.shape)
        return gene, imgs, anns


def fly_collate_fn(batch):
    bgene, bimgs, blabel = zip(*batch)
    size = len(bgene)
    nslice = [x.shape[0] for x in bimgs]
    max_slice = max(nslice)

    pad_imgs = []
    for i in range(size):
        pad_img = np.pad(bimgs[i],
                         [(0, max_slice - nslice[i]), (0, 0),
                          (0, 0), (0, 0)],
                         mode='constant',
                         constant_values=0
                         )
        pad_imgs.append(pad_img)

    return (np.array(bgene),
            torch.from_numpy(np.array(pad_imgs)),
            torch.from_numpy(np.array(blabel)),
            torch.from_numpy(np.array(nslice)))


if __name__ == "__main__":
    pass
