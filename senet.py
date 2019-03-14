#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torch.utils import model_zoo
from pretrainedmodels.models.senet import SENet
from pretrainedmodels.models.senet import SEResNetBottleneck
# from pretrainedmodels.models.senet import SEResNeXtBottleneck

URL = 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth'
# URL = 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth'


class FlySENet(SENet):

    def __init__(self, k=10):
        super(FlySENet, self).__init__(
            SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
            dropout_p=None, inplanes=64, input_3x3=False,
            downsample_kernel_size=1, downsample_padding=0,
            num_classes=1000)
        # super(FlySENet, self).__init__(
        #     SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
        #     dropout_p=None, inplanes=64, input_3x3=False,
        #     downsample_kernel_size=1, downsample_padding=0,
        #     num_classes=1000)

        self.load_state_dict(model_zoo.load_url(URL))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.last_linear = nn.Linear(2048, k, bias=False)

    def forward(self, x):
        fc = super(FlySENet, self).forward(x)
        return torch.sigmoid(fc)
