#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import extractor


class DRAGN(nn.Module):

    def __init__(self):
        super(DRAGN, self).__init__()
        self.fvextractor = extractor.Resnet()

    def forward(self, x):
        pass


if __name__ == "__main__":
    pass
