#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import extractor
import gpustat

FV_DIM = 512


def get_gpu_usage(device=1):
    gpu_stats = gpustat.new_query()
    item = gpu_stats.jsonify()['gpus'][device]
    return item['memory.used'] / item['memory.total']


class L1Aggregator(nn.Module):

    def __init__(self):
        super(L1Aggregator, self).__init__()
        self.q0 = nn.Parameter(torch.ones((1, FV_DIM)))
        self.w = nn.Parameter(torch.ones(FV_DIM))
        self.b = nn.Parameter(torch.zeros(FV_DIM))

    def forward(self, fvs):
        '''single attention block'''
        ek = torch.mm(self.q0, fvs.t())
        ak = torch.nn.Softmax(dim=1)(ek)
        r0 = torch.mm(ak, fvs)
        return r0


class L2Aggregator(nn.Module):

    def __init__(self):
        pass

    def forward(self, x):
        pass


class NAggN(nn.Module):

    def __init__(self, k=10):
        super(NAggN, self).__init__()
        self.fvextractor = extractor.Resnet()
        self.aggregator = L1Aggregator()
        self.proj = nn.Linear(FV_DIM, k)

    def forward(self, x, nslice):
        # print("forward", x.shape, "slice", nslice)
        nsample = x.shape[0]
        nfvs = []
        for s in range(nsample):
            fvs = self.fvextractor(x[s, :nslice[s], :, :])
            # print("fvs", fvs.shape)
            gfv = self.aggregator(fvs)
            # print("gfv", gfv.shape)
            nfvs.append(gfv)
        nfvs = torch.cat(nfvs, dim=0)
        # print("nfvs", nfvs.shape)
        return torch.sigmoid(self.proj(nfvs))


if __name__ == "__main__":
    pass
