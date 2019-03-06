#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import extractor


class DRAggregator(nn.Module):

    def __init__(self):
        super(DRAggregator, self).__init__()
        self.conv0 = nn.Conv2d(3, 1, kernel_size=3, stride=1,
                               padding=1, bias=False)

    def forward(self, fvs):
        ns, h, w = fvs.shape
        if ns == 2:
            return torch.mean(fvs, keepdim=True)
        out = []
        for i in range(ns-3):
            cfv = fvs[i:i+3, :]
            out.append(self.conv0(cfv))
        return torch.concat(out, dim=0)


class DRAGN(nn.Module):

    def __init__(self, k=10, nblock=4):
        super(DRAGN, self).__init__()
        self.fvextractor = extractor.ConvResnet(nblock)
        fvdim = 512 // (2 ** (4 - nblock))
        self.aggregator = DRAggregator()
        self.proj = nn.Linear(fvdim, k)

    def forward(self, x, nslice):
        '''x: nb x ns x c x h x w'''
        nsample = x.shape[0]
        nfvs = []
        for s in range(nsample):
            fvs = self.fvextractor(x[s, :nslice[s], :])
            while True:
                ns, h, w = fvs.shape
                if ns == 1:
                    gfv = fvs
                    break
                else:
                    fvs = self.aggregator(fvs)
            nfvs.append(gfv)
        nfvs = torch.cat(nfvs, dim=0)
        return torch.sigmoid(self.proj(nfvs))


if __name__ == "__main__":
    pass
