#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.nn as nn


class ConvResnet(nn.Module):

    def __init__(self, nblock=4):
        super(ConvResnet, self).__init__()
        origin = torchvision.models.resnet18(pretrained=True)
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.conv0.weight = torch.nn.Parameter(origin.conv1.weight)
        self.bn0 = origin.bn1
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layer0 = nn.Sequential(self.conv0, self.bn0,
                               self.relu0, self.pool0)
        layer1 = origin.layer1
        layer2 = origin.layer2
        layer3 = origin.layer3
        layer4 = origin.layer4

        if nblock == 4:
            self.layers = [layer0, layer1, layer2, layer3, layer4]
        elif nblock == 3:
            self.layers = [layer0, layer1, layer2, layer3]
        elif nblock == 2:
            self.layers = [layer0, layer1, layer2]
        elif nblock == 1:
            self.layers = [layer0, layer1]
        else:
            raise Exception("Invalid nblock", nblock)

        self.features = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.features(x)


class Resnet(nn.Module):

    def __init__(self, nblock=4):
        super(Resnet, self).__init__()
        self.features = ConvResnet(nblock)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        f = self.features(x)
        p = self.pool(f)
        return p.view(p.size(0), -1)


if __name__ == "__main__":
    pass
