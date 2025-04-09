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


from glob import glob
import os
from pathlib import Path
import cv2
import numpy as np


def make_border(img, color=(0, 0, 255), size=3):
    H, W, C = img.shape
    img = img[size : H - size, size : W - size, :]
    border = cv2.copyMakeBorder(
        img,
        top=size,
        bottom=size,
        left=size,
        right=size,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )

    return border

def zoomer(img, bbox, zoom=1.0):
    H, W, C = img.shape
    color = (92, 203, 245, 255)
    top, left, height, width = bbox
    zoom_area = img[top : top + height, left : left + width, :].copy()

    width = int(width * zoom)
    height = int(height * zoom)
    zoomed = cv2.resize(zoom_area, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    img[H - height: H, W - width:W, :] = make_border(zoomed, size=4, color=color).copy()

    return img
