#!/usr/bin/env python
# -*- coding: utf-8 -*-

# merge result from multi stage

import os
import pandas as pd
import numpy as np


def merge_result(model='resnet_si'):

    metrics = ['auc', 'f1_macro', 'f1_micro',
               'sensitivity', 'specificity']

    d = {}
    for m in metrics:
        d[m] = []

    for s in range(1, 7):
        model_dir = os.path.join('modeldir/stage%d' % s, model)
        pth = os.path.join(model_dir, 'metrics.csv')
        if not os.path.exists(pth):
            continue
        df = pd.read_csv(pth, sep=':', header=None, index_col=0)
        for m in metrics:
            d[m].extend(df.loc[m].values)

    outf = os.path.join('result/%s.csv' % model)
    with open(outf, 'w') as f:
        for m in metrics:
            f.write("%s:%.4f\n" % (m, np.mean(d[m])))


if __name__ == "__main__":
    merge_result('resnet_pj')
    # merge_result("resnet_si")
