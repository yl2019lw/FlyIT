#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import extractor


class PostAggregator(nn.Module):

    def __init__(self):
        super(PostAggregator, self).__init__()
        self.conv0 = nn.Conv2d(3, 1, kernel_size=3, stride=1,
                               padding=1, bias=False)

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
                    res = torch.transpose(self.conv0(cfv), 0, 1)
                    out.append(res)
                out = torch.cat(out, dim=0)
                return self.forward(out)


class PostDRAGN(nn.Module):

    def __init__(self, k=10, nblock=4):
        super(PostDRAGN, self).__init__()
        self.fvextractor = extractor.ConvResnet(nblock)
        fvdim = 512 // (2 ** (4 - nblock))
        self.aggregator = PostAggregator()
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


class PreDRAGN(nn.Module):

    def __init__(self, k=10, nblock=4):
        super(PreDRAGN, self).__init__()

    def forward(self, x, nslice):
        pass


if __name__ == "__main__":
    pass
