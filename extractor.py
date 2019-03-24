#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.nn as nn


class ConvResnet(nn.Module):

    def __init__(self, nblock=4, basic='resnet18'):
        super(ConvResnet, self).__init__()
        if basic == 'resnet18':
            origin = torchvision.models.resnet18(pretrained=True)
        elif basic == 'resnet34':
            origin = torchvision.models.resnet34(pretrained=True)
        elif basic == 'resnet50':
            origin = torchvision.models.resnet50(pretrained=True)

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

    def __init__(self, nblock=4, basic='resnet18'):
        super(Resnet, self).__init__()
        self.features = ConvResnet(nblock, basic)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        f = self.features(x)
        p = self.pool(f)
        return p.view(p.size(0), -1)


class SmallFeat(nn.Module):

    def __init__(self, k=10):
        super(SmallFeat, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)


if __name__ == "__main__":
    pass
