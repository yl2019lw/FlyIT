#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import extractor
import sys

sys.setrecursionlimit(10240)


class L1Agg(nn.Module):

    def __init__(self, withbn=False):
        super(L1Agg, self).__init__()
        self.withbn = withbn
        self.conv0 = nn.Conv2d(3, 1, kernel_size=3,
                               stride=1, padding=1)

    def forward(self, x):
        return self.conv0(x)


class L2Agg(nn.Module):

    def __init__(self, withbn=False):
        super(L2Agg, self).__init__()
        self.withbn = withbn
        self.conv0 = nn.Conv2d(3, 3, kernel_size=3,
                               stride=1, padding=1)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 1, kernel_size=3,
                               stride=1, padding=1)
        if self.withbn:
            self.bn0 = nn.BatchNorm2d(3)

    def forward(self, x):
        if self.withbn:
            return self.conv1(self.relu0(self.bn0(self.conv0(x))))
        else:
            return self.conv1(self.relu0(self.conv0(x)))


class L3Agg(nn.Module):

    def __init__(self, withbn=False):
        super(L3Agg, self).__init__()
        self.withbn = withbn
        self.conv0 = nn.Conv2d(3, 3, kernel_size=3,
                               stride=1, padding=1)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3,
                               stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(3, 1, kernel_size=3,
                               stride=1, padding=1)

        if self.withbn:
            self.bn0 = nn.BatchNorm2d(3)
            self.bn1 = nn.BatchNorm2d(3)

    def forward(self, x):
        if self.withbn:
            tmp = self.conv1(self.relu0(self.bn0(self.conv0(x))))
            return self.conv2(self.relu1(self.bn1(tmp)))
        else:
            tmp = self.conv1(self.relu0(self.conv0(x)))
            return self.conv2(self.relu1(tmp))


class PostAggregator(nn.Module):

    def __init__(self, level=2, withbn=False):
        super(PostAggregator, self).__init__()
        if level == 1:
            self.agg = L1Agg(withbn)
        elif level == 2:
            self.agg = L2Agg(withbn)
        else:
            self.agg = L3Agg(withbn)

    def forward(self, fvs):
        while True:
            ns, nc, h, w = fvs.shape
            if ns == 1:
                return fvs
            elif ns == 2:
                return torch.mean(fvs, dim=0, keepdim=True)
            else:
                out = []
                for i in range(ns-2):
                    cfv = torch.transpose(fvs[i:i+3, :], 0, 1)
                    agg = self.agg(cfv)
                    res = torch.transpose(agg, 0, 1)
                    out.append(res)
                out = torch.cat(out, dim=0)
                return self.forward(out)


class PostDRAGN(nn.Module):

    def __init__(self, k=10, nblock=4, agglevel=2, withbn=False):
        super(PostDRAGN, self).__init__()
        self.fvextractor = extractor.ConvResnet(nblock)
        fvdim = 512 // (2 ** (4 - nblock))
        self.aggregator = PostAggregator(agglevel, withbn)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(fvdim, k)

    def forward(self, x, nslice):
        nb, ns, c, h, w = x.shape
        img_count = torch.sum(nslice)
        allimgs = torch.zeros(img_count, c, h, w).cuda()
        cur = 0
        for s in range(nb):
            allimgs[cur:cur+nslice[s]] = x[s, :nslice[s], :]
            cur = cur+nslice[s]

        allfvs = self.fvextractor(allimgs)
        agg_fvs = []
        cur = 0
        for s in range(nb):
            s_fv = allfvs[cur:cur+nslice[s]]
            cur = cur + nslice[s]
            agg_fv = self.aggregator(s_fv)
            agg_fvs.append(agg_fv)

        agg_fvs = torch.cat(agg_fvs, dim=0)
        pool = self.pool(agg_fvs)
        pool = pool.view(pool.size(0), -1)
        return torch.sigmoid(self.proj(pool))


class PreAggregator(nn.Module):

    def __init__(self, level=2, withbn=False):
        super(PreAggregator, self).__init__()
        if level == 1:
            self.agg = L1Agg(withbn)
        elif level == 2:
            self.agg = L2Agg(withbn)
        else:
            self.agg = L3Agg(withbn)

    def forward(self, x, nslice):
        nb, ns, c, h, w = x.shape

        agg_fvs = []
        for s in range(nb):
            s_fv = x[s, :nslice[s], :]
            agg_fv = self.recursive_agg(s_fv)
            agg_fvs.append(agg_fv)

        agg_fvs = torch.cat(agg_fvs, dim=0)
        return agg_fvs

    def recursive_agg(self, fvs):
        while True:
            ns, nc, h, w = fvs.shape
            if ns == 1:
                return fvs
            elif ns == 2:
                return torch.mean(fvs, dim=0, keepdim=True)
            else:
                out = []
                for i in range(ns-2):
                    cfv = torch.transpose(fvs[i:i+3, :], 0, 1)
                    agg = self.agg(cfv)
                    res = torch.transpose(agg, 0, 1)
                    out.append(res)
                out = torch.cat(out, dim=0)
                return self.recursive_agg(out)


class PreDRAGN(nn.Module):

    def __init__(self, k=10, nblock=4, agglevel=2, withbn=False):
        super(PreDRAGN, self).__init__()
        self.fvextractor = extractor.ConvResnet(nblock)
        fvdim = 512 // (2 ** (4 - nblock))
        self.aggregator = PreAggregator(agglevel, withbn)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(fvdim, k)

    def forward(self, x, nslice):
        agg_imgs = self.aggregator(x, nslice)
        agg_fvs = self.fvextractor(agg_imgs)
        pool = self.pool(agg_fvs)
        pool = pool.view(pool.size(0), -1)
        return torch.sigmoid(self.proj(pool))


if __name__ == "__main__":
    pass
