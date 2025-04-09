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


def paste(img, size, bg_color="white", dst=None):
    H, W = size[4], size[5]
    C = min(img.shape)

    if th.is_tensor(img):
        if dst is None:
            if bg_color.lower() == "white":
                dst = th.ones([C, H, W]).cuda().float()
            else:
                dst = th.zeros([C, H, W]).cuda().float()
        dst[:, size[0] : size[1], size[2] : size[3]] = img
        return dst

    # I know, different shapes...
    if dst is None:
        if bg_color.lower() == "white":
            dst = np.ones([H, W, C])
        else:
            dst = np.zeros([H, W, C])

    dst[size[0] : size[1], size[2] : size[3], :] = img
    return dst
