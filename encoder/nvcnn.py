# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2025 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: wojciech.zielonka@tuebingen.mpg.de, wojciech.zielonka@tu-darmstadt.de


import torch.nn as nn
import torch


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu")


class NvCNN(nn.Module):
    def __init__(self, outsize):
        super(NvCNN, self).__init__()
        cnn_output = 11664 # assuming input image H // 3 and W // 3

        # https://arxiv.org/pdf/1609.06536.pdf
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.Conv2d(96, 144, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(144, 144, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(144, 216, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(216, 216, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(216, 324, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(324, 324, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(324, 486, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(486),
            nn.LeakyReLU(),
            nn.Conv2d(486, 486, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(cnn_output, 160),
            nn.LeakyReLU(),
        )

        self.fc.apply(kaiming_leaky_init)
        self.output = nn.Linear(160, outsize)
        with torch.no_grad():
            self.output.weight *= 0.33

        for m in self.cnn:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')


    def forward(self, x):
        features = self.cnn(x)
        params = self.fc(features.view(features.size(0), -1))
        return self.output(params)


class DoubleNvCNN(nn.Module):
    def __init__(self, outsize):
        super(DoubleNvCNN, self).__init__()

        self.geometry = NvCNN(outsize=outsize - 4 - 3)
        self.orientation = NvCNN(outsize=4 + 3)

    def forward(self, x):
        coeffs = self.geometry(x)
        RT = self.orientation(x)
        return coeffs, RT
