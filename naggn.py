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
    '''single attention block'''
    def __init__(self):
        super(L1Aggregator, self).__init__()
        self.q0 = nn.Parameter(torch.ones((1, FV_DIM)))

    def forward(self, fvs):
        '''fvs: ns x nd, r0: 1 x nd'''
        norm = torch.norm(fvs, p=2, dim=1, keepdim=True)
        fvs = torch.div(fvs, norm.expand_as(fvs))
        ek = torch.mm(self.q0, fvs.t())
        ak = torch.nn.Softmax(dim=1)(ek)
        r0 = torch.mm(ak, fvs)
        return r0


class L2Aggregator(nn.Module):
    '''cascaded two attention blocks'''
    def __init__(self):
        super(L2Aggregator, self).__init__()
        self.q0 = nn.Parameter(torch.zeros((1, FV_DIM)))
        self.W = nn.Parameter(torch.randn(FV_DIM, FV_DIM))
        self.b = nn.Parameter(torch.zeros(1, FV_DIM))

    def forward(self, fvs):
        e0k = torch.mm(self.q0, fvs.t())
        a0k = torch.nn.Softmax(dim=1)(e0k)
        r0 = torch.mm(a0k, fvs)

        q1 = torch.tanh(torch.mm(r0, self.W) + self.b)
        e1k = torch.mm(q1, fvs.t())
        a1k = torch.nn.Softmax(dim=1)(e1k)
        r1 = torch.mm(a1k, fvs)

        return r1


class NAggN(nn.Module):

    def __init__(self, k=10):
        super(NAggN, self).__init__()
        self.fvextractor = extractor.Resnet()
        self.aggregator = L1Aggregator()
        self.proj = nn.Linear(FV_DIM, k)

    def forward(self, x, nslice):
        '''x: nb x ns x c x h x w'''
        nsample = x.shape[0]
        nfvs = []
        for s in range(nsample):
            fvs = self.fvextractor(x[s, :nslice[s], :])
            gfv = self.aggregator(fvs)
            nfvs.append(gfv)
        nfvs = torch.cat(nfvs, dim=0)
        return torch.sigmoid(self.proj(nfvs))


if __name__ == "__main__":
    pass
