#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def merge_csv(mlist=[], outfile="out.csv", basedir='modeldir/stage_all'):
    inlist = ['%s/%s/metrics.csv' % (basedir, x) for x in mlist]
    d = {}
    d['loss'] = mlist
    for path in inlist:
        with open(path, 'r') as f:
            for line in f.readlines():
                key, value = line.strip("\n").split(":")
                if key not in d:
                    d[key] = []
                d[key].append(value)

    df = pd.DataFrame(d).set_index('loss')
    df.to_csv(outfile)


def plot_loss(infile='loss.csv', outfile='loss.eps'):
    df = pd.read_csv(infile).set_index('loss')
    methods = np.array(df.index)
    # metrics = np.array(df.columns)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bar_width = 0.2
    opacity = 0.8
    # colors = ['b', 'g', 'r', 'c', 'm', 'k']
    colors = ['#3399CC', '#33CC99', '#660066', '#996633', '#CC3366', '#CCFF66']

    # pos = np.arange(len(metrics))
    pos = np.array([0.4, 2.0, 3.6, 5.2, 6.8])
    for i, m in enumerate(methods):
        plt.bar(pos + i*bar_width, df.loc[m], bar_width,
                alpha=opacity,  color=colors[i],
                label=m)

    legend_labels = [
        r'$BCE$', r'$FEC(\alpha=1)$',
        r'$FEC(\alpha=2)$', r'$FEC(\alpha=3)$',
        r'$FEC(\alpha=4)$']
    metrics_labels = [
        'auc', 'f1_macro', 'f1_micro',
        'sensitivity', 'specificity']

    # plt.title("test")
    plt.xticks(pos+0.4, metrics_labels)
    ax.tick_params(axis='x', length=0, pad=8)

    ax.set_xlim(left=-0.2, right=9.0)
    ax.set_ylim(bottom=0.0, top=1.15)
    leg = plt.legend(labels=legend_labels, loc='best', ncol=5,
                     mode='expand', shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.4)
    plt.tight_layout()
    # plt.legend(labels=legend_labels, title=r'Different Losses',
    #            loc='upper left', bbox_to_anchor=(1, 1))
    # plt.tight_layout(rect=[0, 0, 0.85, 1.0])
    # plt.show()
    plt.savefig(outfile)


def test():
    methods = ['resnet18b4_pj_k10', 'resnet18b4_pj_k20', 'resnet18b4_pj_k30']
    for m in methods:
        mlist = [m + x
                 for x in ['', '_fec1', '_fec2', '_fec3', '_fec4']]
        merge_csv(mlist, "result/stage_all/%s.csv" % m,
                  basedir='modeldir/stage_all/')

        merge_csv(mlist, "result/stage_all/seq_pj/%s.csv" % m,
                  basedir='modeldir/stage_all/seq_pj/')


if __name__ == "__main__":
    # test()
    plot_loss('result/stage_all/resnet18b4_pj_k10.csv',
              'resnet18b4_pj_k10.eps')

    plot_loss('result/stage_all/resnet18b4_pj_k20.csv',
              'resnet18b4_pj_k20.eps')

    plot_loss('result/stage_all/resnet18b4_pj_k30.csv',
              'resnet18b4_pj_k30.eps')
    # plot_loss('result/stage_all/seq_pj/resnet18b4_pj_k20.csv',
    #           'resnet18b4_seq_pj_k20.eps')
