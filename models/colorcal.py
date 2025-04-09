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


from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch as th


def scale_hook(grad: Optional[th.Tensor], scale: float) -> Optional[th.Tensor]:
    if grad is not None:
        grad = grad * scale
    return grad


class ColorCalibration(th.nn.Module):
    def __init__(self, cameras, identity_camera=None) -> None:
        super().__init__()

        if identity_camera is None or identity_camera not in cameras:
            identity_camera = cameras[0]

        self.n_cameras = len(cameras)
        self.identity_camera = identity_camera
        self.cameras = cameras
        self.identity_idx = cameras.index(identity_camera)
        self.corrections = th.nn.Parameter(
            th.FloatTensor([[1, 1, 1, 0, 0, 0]]).expand(self.n_cameras, -1).cuda().requires_grad_(True)
        )

        self.cam2index = {}
        for i, cam in enumerate(cameras):
            self.cam2index[cam] = i

    def forward(self, rbg, cam_name) -> th.Tensor:
        cam_idx = self.cam2index[cam_name]
        params = self.corrections[cam_idx]

        if self.identity_camera == cam_name:
            return rbg

        w, b = params[:3], params[3:]

        if len(rbg.shape) == 3:
            out = rbg * w[:, None, None] + b[:, None, None]
        else:
            out = rbg * w + b

        col_lrscalet = 1e-1
        if self.training and params.requires_grad:
            params.register_hook(lambda g: scale_hook(g, col_lrscalet))

        return out
