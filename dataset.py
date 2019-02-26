#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from collections import Counter


def load_by_stage(stage=1):
    img = pd.read_csv('data/standard_images.csv')
    img = img[img['stage'] == stage]

    ann = pd.read_csv('data/standard_annotations.csv')
    ann = ann[ann['stage'] == stage]

    i_gene = img['gene'].unique()
    a_gene = ann['gene'].unique()

    gene = np.intersect1d(i_gene, a_gene)

    d = {}
    for g in gene:
        d[g] = {}
        gene_imgs = []
        for item in img[img['gene'] == g]['image_url'].values:
            gene_imgs += list(eval(item))

        gene_annotations = []
        for item in ann[ann['gene'] == g]['annotation'].values:
            gene_annotations += list(eval(item))

        d[g]['img'] = gene_imgs
        d[g]['ann'] = gene_annotations

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


class DrosophilaDataset(Dataset):

    def __init__(self, mode='train', stage=2):
        self.db, self.top_cv = filter_top_cv(load_by_stage(stage))
        self.nclass = len(self.top_cv)
        genes = list(self.db.keys())
        tl = int(len(genes) * 0.6)
        vl = int(len(genes) * 0.8)
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

    return (np.array(bgene), np.array(pad_imgs),
            np.array(blabel), np.array(nslice))


if __name__ == "__main__":
    pass
