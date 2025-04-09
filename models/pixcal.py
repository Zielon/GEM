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


import torch as th
import torch.nn.functional as F


class CameraPixelBias(th.nn.Module):
    def __init__(self, image_height, image_width, ds_rate, cameras) -> None:
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.cameras = cameras
        self.n_cameras = len(cameras)

        bias = th.zeros((self.n_cameras, 1, image_width // ds_rate, image_height // ds_rate), dtype=th.float32)
        self.register_parameter("bias", th.nn.Parameter(bias))

    def forward(self, idxs: th.Tensor):
        bias_up = F.interpolate(self.bias[idxs], size=(self.image_height, self.image_width), mode="bilinear")
        return bias_up
