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


import json

from glob import glob
from pathlib import Path

import torch as th
from pytorch3d.transforms import axis_angle_to_matrix
import numpy as np
from tqdm import tqdm
import trimesh

from .flame import FLAME


def get_flame_extra_faces():
    return np.array(
        [
            [1573, 1572, 1860],
            [1742, 1862, 1572],
            [1830, 1739, 1665],
            [2857, 2862, 2730],
            [2708, 2857, 2730],
            [1862, 1742, 1739],
            [1830, 1862, 1739],
            [1852, 1835, 1666],
            [1835, 1665, 1666],
            [2862, 2861, 2731],
            [1747, 1742, 1594],
            [3497, 1852, 3514],
            [1595, 1747, 1594],
            [1746, 1747, 1595],
            [1742, 1572, 1594],
            [2941, 3514, 2783],
            [2708, 2945, 2857],
            [2941, 3497, 3514],
            [1852, 1666, 3514],
            [2930, 2933, 2782],
            [2933, 2941, 2783],
            [2862, 2731, 2730],
            [2945, 2930, 2854],
            [1835, 1830, 1665],
            [2857, 2945, 2854],
            [1572, 1862, 1860],
            [2854, 2930, 2782],
            [2708, 2709, 2943],
            [2782, 2933, 2783],
            [2708, 2943, 2945],
        ]
    )


flame = FLAME()

faces = flame.faces_tensor.cpu().numpy()
faces = np.concatenate([faces, get_flame_extra_faces()])

def to_mesh(params, use_delta=True, R=None, T=None):
    neck_pose = th.from_numpy(params["neck_pose"]).float()
    jaw_pose = th.from_numpy(params["jaw_pose"]).float()
    eyes_pose = th.from_numpy(params["eyes_pose"]).float()
    shape = th.from_numpy(params["shape"]).float()
    expr = th.from_numpy(params["expr"]).float()
    if R is None:
        R = th.zeros([1, 3])
    pose = th.cat([R, jaw_pose], dim=-1).float()
    static_offset = None
    if use_delta:
        static_offset = th.from_numpy(params["static_offset"])[:, :5023, :].float()

    vertices, J, A, W = flame(
        shape_params=shape, 
        expression_params=expr, 
        pose_params=pose, 
        neck_pose=neck_pose, 
        eye_pose=eyes_pose, 
        delta=static_offset,
        transl=T
    )

    return trimesh.Trimesh(vertices[0].numpy(), faces, process=False), J, A, W


def to_lbs(params, path):
    mesh, J, A, W = to_mesh(params)
    # mesh.export("mesh_neck.ply")
    vertices = th.from_numpy(mesh.vertices).float()[None]

    lbs = {
        "J": J,
        "A": A,
        "W": W
    }

    th.save(lbs, path)

    A[:, 1, ...] = th.linalg.inv(A[:, 1, ...])

    W[:, :, 0] = 0 
    W[:, :, 1] = 1
    W[:, :, 2] = 0 
    W[:, :, 3] = 0 
    W[:, :, 4] = 0 

    T = th.matmul(W, A.view(1, 5, 16)).view(1, -1, 4, 4)

    homogen_coord = th.ones([1, vertices.shape[1], 1])
    v_posed_homo = th.cat([vertices, homogen_coord], dim=2)
    v_homo = th.matmul(T, th.unsqueeze(v_posed_homo, dim=-1))

    vertices = v_homo[:, :, :3, 0]

    mesh.vertices = vertices.numpy()[0]

    # mesh.export("mesh_no_neck.ply")


def to_dict(npz):
    params = {}
    for key in ["neck_pose", "jaw_pose", "eyes_pose", "shape", "expr", "static_offset", "rotation", "translation"]:
        params[key] = npz[key]
    return params


def to_canonical(params):
    neck_pose = th.from_numpy(params["neck_pose"]).float() * 0
    jaw_pose = th.from_numpy(params["jaw_pose"]).float() * 0
    eyes_pose = th.from_numpy(params["eyes_pose"]).float() * 0
    shape = th.from_numpy(params["shape"]).float()
    expr = th.from_numpy(params["expr"]).float() * 0

    angle = np.radians(20)
    jaw_pose[:, 0] = angle

    R = th.zeros([1, 3])
    T = th.zeros([1, 3])
    pose = th.cat([R, jaw_pose], dim=-1).float()
    static_offset = th.from_numpy(params["static_offset"])[:, :5023, :].float()

    vertices = flame(
        shape_params=shape, 
        expression_params=expr, 
        pose_params=pose, 
        neck_pose=neck_pose, 
        eye_pose=eyes_pose, 
        delta=static_offset,
        transl=T
    )[0]

    return trimesh.Trimesh(vertices[0].numpy(), faces, process=False)
