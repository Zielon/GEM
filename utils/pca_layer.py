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


import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class PCALayer(nn.Module):
    def __init__(self, components, mean, variance, n_components=100) -> None:
        super().__init__()
        logger.info(f"Loaded PCALayer with {n_components} components")
        self.n_compoents = n_components
        self.register_buffer("components", th.from_numpy(components).float().cuda()[: self.n_compoents, :])
        self.register_buffer("mean", th.from_numpy(mean).float().cuda())
        self.register_buffer("variance", th.from_numpy(variance).float().cuda())
        self.register_buffer("scale", th.from_numpy(np.sqrt(variance)).float().cuda()[: self.n_compoents])

    def transform(self, vertices):
        X = vertices - self.mean
        coeff = th.matmul(X, self.components.T)
        coeff = coeff / self.scale
        return coeff

    def inverse_transform(self, coeff):
        vertices = th.matmul(coeff, self.scale[:, None] * self.components) + self.mean
        return vertices.reshape(-1, 3)

    def forward(self, vertices):
        coeffs = self.transform(vertices.reshape(-1))
        nearest = self.inverse_transform(coeffs)

        return nearest
