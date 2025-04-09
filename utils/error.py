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
from gaussians.losses import l1_loss, ssim
from lpips import lpips

from utils.text import write_text


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * th.log10(1.0 / th.sqrt(mse))


def dist_to_rgb_jet(errors, min_dist=0.0, max_dist=1.0):
    import matplotlib as mpl
    import matplotlib.cm as cm

    norm = mpl.colors.Normalize(vmin=min_dist, vmax=max_dist)
    cmap = cm.get_cmap(name="jet")
    colormapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    return colormapper.to_rgba(errors)


def dist_to_rgb(errors):
    h, w, d = errors.shape
    scale = 1.0
    errors = np.clip(0, scale, errors)
    errors = errors.reshape(h * w)
    heat = dist_to_rgb_jet(errors, 0, scale)[:, 0:3]
    heat = heat.reshape(h, w, 3)
    heat = np.minimum(np.maximum(heat * 255, 0), 255).astype(np.uint8)
    return heat


def compute_heatmap(target, fake):
    p = psnr(fake, target).mean().item()

    target = target.permute(1, 2, 0).cpu().numpy()
    fake = fake.permute(1, 2, 0).cpu().numpy()

    errors = np.linalg.norm((target - fake), axis=2, keepdims=True, ord=2)
    heat = dist_to_rgb(errors) / 255.0
    heat = heat.astype(np.float32)

    heatmap = write_text(heat, f"{p:.3f} (dB)", fontColor=(1, 1, 1))

    return heatmap


def compute_errors(target, fake):
    psnr_error = psnr(fake, target).mean().item()
    lpips_error = lpips(fake, target).mean().item()
    l1 = l1_loss(target, fake).mean().item()
    ss = ssim(target, fake).mean().item()

    return {"psnr": psnr_error, "lpips": lpips_error, "l1": l1, "ssim": ss}
