#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset import SIDataset
from dataset import StratifySIDataset
from dataset import load_by_stage
from dataset import filter_top_cv
from collections import Counter


def data_stat(stage=1, k=10):
    print("-------data stat for stage %d-------" % stage)
    d = load_by_stage(stage=stage)
    d, top_cv = filter_top_cv(d, k)
    genes = list(d.keys())
    imgs = []
    all_cv = []
    for g in genes:
        imgs.extend(d[g]['img'])
        all_cv += d[g]['ann']
    print("stage", stage, "genes", len(genes), "imgs", len(imgs))
    count = Counter(all_cv)
    print(count)
    print("\n")


def dataset_tvt_stat(train, val, test):

    def count_gene(genes):
        gene_cv = []
        img_cv = []
        for g in genes:
            gene_cv += d[g]['ann']
            for img in d[g]['ann']:
                img_cv += d[g]['ann']
        gene_count = Counter(gene_cv)
        img_count = Counter(img_cv)
        return gene_count, img_count

    d = train.db
    train_gc, train_ic = count_gene(train.genes)
    val_gc, val_ic = count_gene(val.genes)
    test_gc, test_ic = count_gene(test.genes)

    top_cv = train.top_cv

    print("--------gene count---------")
    for cv in top_cv:
        print("cv:%s\n \t  train:%d, val:%d, test:%d" % (
            cv, train_gc[cv], val_gc[cv], test_gc[cv]))

    print('\n')
    print("--------img count---------")
    for cv in top_cv:
        print("cv:%s\n \t train:%d, val:%d, test:%d" % (
            cv, train_ic[cv], val_ic[cv], test_ic[cv]))


def sidataset_tvt_stat(s=2, k=10):
    print("---SIDataset train/val/test stat stage:%d, k:%d---" % (s, k))
    data_stat(stage=s, k=k)

    val_index = 4
    train = SIDataset(mode='train', stage=s, k=k, val_index=val_index)
    val = SIDataset(mode='val', stage=s, k=k, val_index=val_index)
    test = SIDataset(mode='test', stage=s, k=k, val_index=val_index)

    dataset_tvt_stat(train, val, test)


def stratify_tvt_stat(s=2, k=10):
    print("---StratifyDataset train/val/test stat stage:%d, k:%d---" % (s, k))
    data_stat(stage=s, k=k)

    train = StratifySIDataset(mode='train', stage=s, k=k)
    val = StratifySIDataset(mode='val', stage=s, k=k)
    test = StratifySIDataset(mode='test', stage=s, k=k)

    dataset_tvt_stat(train, val, test)


if __name__ == "__main__":
    # for s in list(range(1, 7)):
    #     data_stat(s, k=20)

    # sidataset_tvt_stat()

    stratify_tvt_stat()
