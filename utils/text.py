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


import cv2
import numpy as np
import torch


def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale / 10, thickness=1)
        new_width = textSize[0][0]
        if new_width <= width:
            return scale / 10
    return 1


def put_text(img, text, pos=(10, 30), fontScale=1, thickness=1, fontColor=(1, 1, 1)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pos, font, fontScale, fontColor, thickness, cv2.LINE_AA)
    return img


def write_text(img, text, fontColor=(1, 1, 1), thickness=2, bottom=False, X=15, set_W=None):
    convert_back = False
    device = "cpu"
    if torch.is_tensor(img):
        device = img.device
        convert_back = True
        img = img.permute(1, 2, 0).cpu().numpy()
    img = np.ascontiguousarray(img).astype(np.float32)
    H, W, C = img.shape
    font_scale = get_optimal_font_scale(" " * 25, W if set_W is None else set_W)
    Y = int(font_scale * 30) if not bottom else H - int(font_scale * 15)

    textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    textX = int((img.shape[1] - textsize[0]) / 2)
    textY = int((img.shape[0] + textsize[1]) / 2)

    img = put_text(
        img,
        text,
        fontScale=font_scale,
        thickness=thickness,
        fontColor=fontColor,
        pos=(X, Y),
    )

    if convert_back:
        return torch.from_numpy(img).permute(2, 0, 1).to(device)

    return img
