#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
from torch.utils import model_zoo
from pretrainedmodels.models.senet import SENet
from pretrainedmodels.models.senet import SEResNetBottleneck
# from pretrainedmodels.models.senet import SEResNeXtBottleneck
import extractor

URL = 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth'
# URL = 'http://data.lip6.fr/\
#        cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth'


class SiNet(nn.Module):

    def __init__(self, nblock=4, k=10):
        super(SiNet, self).__init__()
        self.fvextractor = extractor.Resnet(nblock)
        fvdim = 512 // (2 ** (4 - nblock))
        self.proj = nn.Linear(fvdim, k)

    def forward(self, x):
        fv = self.fvextractor(x)
        return torch.sigmoid(self.proj(fv))


class TinyNet(nn.Module):

    def __init__(self, k=10):
        super(TinyNet, self).__init__()
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
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((8, 20))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 20, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, k),
        )

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
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return torch.sigmoid(x)


class SmallNet(nn.Module):

    def __init__(self, k=10):
        super(SmallNet, self).__init__()
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
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 10))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 10, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, k),
        )

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
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return torch.sigmoid(x)


class VggNet(nn.Module):
    def __init__(self, k=10):
        super(VggNet, self).__init__()
        self.model = torchvision.models.vgg11_bn(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d((4, 10))
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 10, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(1024, k),
        )

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.model.classifier(x)
        return torch.sigmoid(x)


class Resnet50(nn.Module):

    def __init__(self, k=10):
        super(Resnet50, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model.fc = nn.Linear(2048, k)

    def forward(self, x):
        return torch.sigmoid(self.model(x))


class Resnet101(nn.Module):

    def __init__(self, k=10):
        super(Resnet101, self).__init__()
        self.model = torchvision.models.resnet101(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model.fc = nn.Linear(2048, k)

    def forward(self, x):
        return torch.sigmoid(self.model(x))


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


if __name__ == "__main__":
    pass
