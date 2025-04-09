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


from collections import namedtuple
import torch as th
import torch.nn.functional as F
import trimesh


Mesh = namedtuple("Mesh", "v f")


def to_trimesh(mesh):
    v = mesh.v.cpu().numpy()
    f = mesh.f.cpu().numpy()

    if len(v) == 3:
        v = v[0]
    if len(f) == 3:
        f = f[0]

    return trimesh.Trimesh(v, f, process=False)


def interpolate(t, size):
    return F.interpolate(t[None], size, mode="bilinear")[0]


def flatten(xyz):
    B, C, H, W = xyz.shape
    # return xyz.permute(0, 2, 3, 1).reshape(-1, C).contiguous().clone()

    w, h = th.meshgrid(
        th.arange(H, device="cuda"),
        th.arange(W, device="cuda"),
        indexing="xy",
    )

    idsH = th.flatten(h)
    idsW = th.flatten(w)

    extracted = xyz[:, :, idsH, idsW][0].permute(1, 0)

    return extracted
