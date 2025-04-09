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
import math
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from gaussians.cameras import batch_to_camera
from utils.image import paste

bg_colors = {
    "white": th.tensor([1, 1, 1], dtype=th.float32, device="cuda").float(),
    "black": th.tensor([0, 0, 0], dtype=th.float32, device="cuda").float(),
}

# We need to ad
bg_maps = {
    "white": None,
    "black": None,
    "random": None
}


def paste(img, crop):
    left_w, right_w, top_h, bottom_h, W, H = crop[0], crop[1], crop[2], crop[3], int(crop[4]), int(crop[5])
    if left_w > right_w:
        img = img[:, :, :W]
    else:
        img = img[:, :, -W:]
    if top_h > bottom_h:
        img = img[:, :H, :]
    else:
        img = img[:, -H:, :]

    return img


def pad_image(img, crop, h, w):
    left_w, right_w, top_h, bottom_h, W, H = crop[0], crop[1], crop[2], crop[3], crop[4], crop[5]
    left, right, up, bottom = 0, 0, 0, 0
    dx = int(abs(w - W))
    dy = int(abs(H - h))
    if left_w > right_w:
        right = dx
    else:
        left = dx
    if top_h > bottom_h:
        bottom = dy
    else:
        up = dy

    padded = F.pad(img, (left, right, up, bottom, 0, 0), "constant", 0)

    return padded


def render(batch, pkg, bg_color="black", training=True):
    viewpoint_camera = batch_to_camera(batch)

    # Set up background color
    crop = batch["crop"].cpu().numpy()

    if bg_color != "random":
        color = bg_colors[bg_color]
    else:
        color = th.randn([3]).cuda()

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pkg["sh_degree"] if "sh_degree" in pkg else 0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    shs, colors_precomp = None, None
    if "shs" in pkg:
        shs = pkg["shs"]
    else:
        colors_precomp = pkg["colors_precomp"].contiguous()

    means3D = pkg["means3D"].contiguous()
    opacities = pkg["opacity"].contiguous()
    rotations = pkg["rotation"].contiguous()
    scales = pkg["scales"].contiguous()

    screenspace_points = th.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0

    try:
        screenspace_points.retain_grad()
    except:
        pass

    means2D = screenspace_points

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rasterizer.train(mode=training)

    rendered_image, radii, _, alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    # depth = paste(depth, crop)

    return {
        # "depth": depth,
        "render": paste(rendered_image, crop),
        "alpha": paste(alpha, crop),
        # "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        # "radii": radii,
        "bg_color": color
    }
