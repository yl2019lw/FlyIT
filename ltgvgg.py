#!/usr/bin/env python
# -*- coding: utf-8 -*-

# code copy from ltg

import torchvision
import torch.nn as nn
import torch.nn.functional as F


class LtgVgg(nn.Module):

    def __init__(self):
        super(LtgVgg, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg16 = nn.Sequential(
            vgg16.features[0],
            vgg16.features[1],
            vgg16.features[2],
            vgg16.features[3],
            vgg16.features[4],
            vgg16.features[5],
            vgg16.features[6],
            vgg16.features[7],
            vgg16.features[8],
            vgg16.features[9],
            vgg16.features[10],
            vgg16.features[11],
            vgg16.features[12],
            vgg16.features[13],
            vgg16.features[14],
            vgg16.features[15],
            vgg16.features[16],
            vgg16.features[17],
            vgg16.features[18],
            vgg16.features[19],
            vgg16.features[20]
        )

        self.addiction = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),  # (batch_size,64,8,20),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),  # (batch_size,32,4,10)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2)  # batch_size,16,2,5
        )
        self.fc = nn.Sequential(
            nn.Linear(10240, 1024),
            # nn.LeakyReLU(),
            nn.Linear(1024, 128),
            # nn.LeakyReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.vgg16(x)
        x = self.addiction(x)
        x = x.view(-1, 10240)
        x = self.fc(x)

        x = F.sigmoid(x)
        return x


if __name__ == "__main__":
    pass
