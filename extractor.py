#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.nn as nn


class ConvResnet(nn.Module):

    def __init__(self):
        super(ConvResnet, self).__init__()
        origin = torchvision.models.resnet18(pretrained=True)
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.conv0.weight = torch.nn.Parameter(origin.conv1.weight)
        self.bn0 = origin.bn1
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer0 = nn.Sequential(self.conv0, self.bn0,
                                    self.relu0, self.pool0)
        self.layer1 = origin.layer1
        self.layer2 = origin.layer2
        self.layer3 = origin.layer3
        self.layer4 = origin.layer4

        self.features = nn.Sequential(self.layer0, self.layer1, self.layer2,
                                      self.layer3, self.layer4)

    def forward(self, x):
        return self.features(x)


class Resnet(nn.Module):

    def __init__(self):
        super(Resnet, self).__init__()
        self.features = ConvResnet()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        f = self.features(x)
        p = self.pool(f)
        return p.view(p.size(0), -1)


if __name__ == "__main__":
    pass
