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


def merge_pj():
    methods = ['resnet18b4_pj_k10', 'resnet18b4_pj_k20', 'resnet18b4_pj_k30']
    for m in methods:
        mlist = [m + x
                 for x in ['', '_fec1', '_fec2', '_fec3', '_fec4']]
        merge_csv(mlist, "result/stage_all/%s.csv" % m,
                  basedir='modeldir/stage_all/')

        merge_csv(mlist, "result/stage_all/seq_pj/%s.csv" % m,
                  basedir='modeldir/stage_all/seq_pj/')


def merge_si():
    methods = ['resnet18b4_si_k10', 'resnet18b4_si_k20', 'resnet18b4_si_k30']
    for m in methods:
        mlist = [m + x
                 for x in ['', '_fec1', '_fec2', '_fec3', '_fec4']]
        merge_csv(mlist, "result/stage_all/%s.csv" % m,
                  basedir='modeldir/stage_all/')

        merge_csv(mlist, "result/stage_all/seq_si/%s.csv" % m,
                  basedir='modeldir/stage_all/seq_si/')


def plot_all_in_one():

    csvfile = 'result/stage_all/all.csv'
    df = pd.read_csv(csvfile)
    grouped = df.groupby(['method'])

    methods = ['bce', 'fec1', 'fec2', 'fec3', 'fec4']
    methods_display = [
        r'$BCE$', r'$FEC(\alpha=1)$',
        r'$FEC(\alpha=2)$', r'$FEC(\alpha=3)$',
        r'$FEC(\alpha=4)$']

    colors = ['b', 'c', 'g', 'y', 'r']
    metrics = ['auc', 'f1_macro', 'f1_micro', 'sensitivity', 'specificity']
    metrics_display = ['auc', 'f1_macro', 'f1_micro',
                       'sensitivity', 'specificity']

    yranges = [[0.85, 0.95], [0.45, 0.7], [0.5, 0.7],
               [0.4, 0.7], [0.9, 1.0]]
    # x_ticks = np.arange(3)
    x_ticks = np.array([0.2, 1.4, 2.6])

    def subplot(metric, sub_idx, handles, ylim):
        # width = 0.15
        width = 0.2
        ax = plt.subplot(1, 5, sub_idx)
        for idx, (mtd, color) in enumerate(zip(methods, colors)):
            data = grouped.get_group(mtd)[metric]

            handle = ax.bar(x_ticks + (width * idx), data, width,
                            zorder=3, edgecolor='k')
            handles.append(handle)

            ax.set_ylim(ylim)

            # ax.set_title(metrics_display[sub_idx - 1], y=1.03)
            ax.set_title(metrics_display[sub_idx - 1], y=1.0)
            ax.set_xticks(x_ticks + width * (len(methods) - 1) / 2)
            ax.set_xticklabels(['D1', 'D2', 'D3'])

            ax.tick_params(labelcolor='k', color='k')
            for spine in ax.spines.values():
                spine.set_edgecolor('k')

            ax.set_facecolor('w')
            # ax.grid(True, axis='y', color='k', alpha=0.3, zorder=0)

    plt.style.use('ggplot')

    for idx, (metric, ylim) in enumerate(zip(metrics, yranges)):
        handles = []
        subplot(metric, idx + 1, handles, ylim)

    legend = plt.figlegend(handles, methods_display,
                           loc=(0.0, 0.9),
                           fontsize=10,
                           ncol=5, mode='expand',
                           shadow=True, fancybox=True)
    # legend = plt.figlegend(handles, methods_display,
    #                        loc=(0.85, 0.6),
    #                        fontsize=10,
    #                        shadow=True, fancybox=True)

    legend.get_frame().set_facecolor('w')
    legend.get_frame().set_edgecolor('k')

    plt.subplots_adjust(top=0.8, wspace=0.3)
    # plt.subplots_adjust(hspace=0.4)
    # plt.gcf().set_size_inches(15, 6)
    plt.gcf().set_size_inches(24, 4)
    plt.tight_layout()

    plt.savefig('comparison.eps', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # test()
    # plot_loss('result/stage_all/seq_pj/resnet18b4_pj_k20.csv',
    #           'resnet18b4_seq_pj_k20.eps')

    # plot_loss('result/stage_all/resnet18b4_pj_k10.csv',
    #           'resnet18b4_pj_k10.eps')

    # plot_loss('result/stage_all/resnet18b4_pj_k20.csv',
    #           'resnet18b4_pj_k20.eps')

    # plot_loss('result/stage_all/resnet18b4_pj_k30.csv',
    #           'resnet18b4_pj_k30.eps')

    # merge_si()
    # plot_loss('result/stage_all/resnet18b4_si_k10.csv',
    #           'resnet18b4_si_k10.eps')

    # plot_loss('result/stage_all/resnet18b4_si_k20.csv',
    #           'resnet18b4_si_k20.eps')

    # plot_loss('result/stage_all/resnet18b4_si_k30.csv',
    #           'resnet18b4_si_k30.eps')
    plot_all_in_one()
