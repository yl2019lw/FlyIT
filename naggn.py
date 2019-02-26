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


class NAggN(nn.Module):

    def __init__(self):
        super(NAggN, self).__init__()
        self.fvextractor = extractor.Resnet()
        self.q0 = nn.Parameter(torch.ones((1, FV_DIM)))
        self.w = nn.Parameter(torch.ones(FV_DIM))
        self.b = nn.Parameter(torch.zeros(FV_DIM))
        self.proj = nn.Linear(FV_DIM, 1)

    def forward(self, x, nslice):
        print("forward", x.shape, nslice)
        nsample = x.shape[0]
        nfvs = []
        for s in range(nsample):
            fvs = self.extract(x[s, :nslice[s], :, :])
            print("fvs", fvs.shape)
            gfv = self.aggregate(fvs)
            print("gfv", gfv.shape)
            nfvs.append(gfv)
        nfvs = torch.cat(nfvs, dim=0)
        print("nfvs", nfvs.shape)
        return torch.sigmoid(self.proj(nfvs))

    def extract(self, imgs):
        '''extract fv for a single sid'''
        # unsqueeze to b x c x h x w
        cimgs = torch.unsqueeze(imgs, dim=1)
        fvs = self.fvextractor(cimgs)
        return fvs

    def aggregate(self, fvs):
        '''aggregate fvs for a single sid'''
        return self.s_aggregate(fvs)

    def s_aggregate(self, fvs):
        '''single attention block'''
        ek = torch.mm(self.q0, fvs.t())
        print("ek", ek.shape)
        ak = torch.nn.Softmax(dim=1)(ek)
        print("ak", ak.shape, torch.sum(ak, dim=1))
        r0 = torch.mm(ak, fvs)
        return r0

    def c_aggregate(self, fvs):
        '''cascaded two attention blocks'''
        ek = torch.mm(self.q0, fvs.t())
        print("ek", ek.shape)
        ak = torch.nn.Softmax()(ek)
        print("ak", ak.shape)
        r0 = torch.mm(ak, fvs)
        print("r0", r0.shape)
        return r0


if __name__ == "__main__":
    pass
