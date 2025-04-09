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


from enum import Enum
import json
import math
import os

import cv2
import numpy as np
import torch.utils.data
from data.utils import linear2color_corr
from utils.geometry import AttrDict, compute_v2uv


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


class DatasetMode(Enum):
    test = "test"
    validation = "validation"
    train = "train"
    video = "video"


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, config, custom_camera_list=None, mode=DatasetMode.train):
        data = config.data
        self.config = config
        self.imagepath = data.image
        self.maskpath = data.mask
        self.meshpath = data.mesh
        self.ds_rate = config.ds_rate
        self.gamma_space = False
        self.frame_list = []
        self.camera_ids = {}
        self.mesh_topology = None
        self.extra_topology = None
        self.mode = mode
        self.custom_camera_list = custom_camera_list
        self.include_lbs = False
        self.reenactment_dataset = False

    def __len__(self):
        return len(self.frame_list)

    def read_image(self, path, interpolation=cv2.INTER_CUBIC, fn=None):
        if not os.path.exists(path):
            raise FileNotFoundError("File not found: {}".format(path))
        if fn is None:
            image = cv2.imread(path)
        else:
            image = fn(path)
        H, W, C = image.shape
        dim = (W // self.ds_rate, H // self.ds_rate)
        image = cv2.resize(image, dim, interpolation=interpolation)
        image = np.transpose(image[..., ::-1].astype(np.float32), axes=(2, 0, 1))

        return image

    def homogenization(self, R, t):
        I = np.eye(4)
        I[:3, :3] = R
        I[:3, 3] = t
        return I

    def transform_cameras(self, root_Rt, Rt):
        R_root = root_Rt[:3, :3]
        t_root = root_Rt[:3, 3] * 0.001

        R_C = Rt[:3, :3]
        t_C = Rt[:3, 3] * 0.001

        A = self.homogenization(R_C, t_C)
        B = self.homogenization(R_root, t_root)
        w2c = A @ B

        return w2c

    def until_valid(self, frame_pkg):
        sentnum, frame, cam = frame_pkg
        path = "{}/{}/{}/{}.png".format(self.imagepath, sentnum, cam, frame)
        if os.path.exists(path):
            return False
        return True

    def get_topology(self):
        topology = AttrDict(
            dict(
                vi=self.mesh_topology["vert_ids"].astype(np.int64),
                vt=self.mesh_topology["uvs"].astype(np.float32),
                vti=self.mesh_topology["uv_ids"].astype(np.int64),
                v=self.mesh_topology["verts"].astype(np.float32),
            )
        )

        topology.v2uv = compute_v2uv(topology.vi, topology.vti)

        return topology

    def get(self, idx):
        raise NotImplementedError()

    def __getitem__(self, idx):
        pkg = self.get(idx)

        K = pkg.K
        Rt = pkg.Rt
        cam_id = pkg.cam_id
        image = pkg.image
        alpha = pkg.alpha
        verts = pkg.verts
        root_Rt = pkg.root_Rt
        cam = pkg.cam
        frame = pkg.frame
        sentnum = pkg.sentnum

        C, H, W = image.shape

        K[0, 2] /= self.ds_rate
        K[1, 2] /= self.ds_rate
        K[0, 0] /= self.ds_rate
        K[1, 1] /= self.ds_rate

        fx = float(K[0][0])
        fy = float(K[1][1])

        left_w, right_w = K[0, 2], W - K[0, 2]
        top_h, bottom_h = K[1, 2], H - K[1, 2]
        cx = max(left_w, right_w)
        cy = max(top_h, bottom_h)
        w = int(2 * cx)
        h = int(2 * cy)

        crop_params = np.array([left_w, right_w, top_h, bottom_h, W, H])

        R = np.transpose(Rt[:3, :3])
        T = Rt[:3, 3]

        if self.gamma_space:
            image = linear2color_corr(image, dim=0).astype(np.float32)

        final = {
            "cam_idx": cam,
            "frame": frame,
            "exp": sentnum,
            "cam": cam_id,
            "image": image,
            "alpha": alpha,
            "crop": crop_params,
            "R": R,
            "T": T,
            "cam_RT": Rt.astype(np.float32),
            "root_RT": root_Rt.astype(np.float32),
            "K": K.astype(np.float32),
            "fl_x": fx,
            "fl_y": fy,
            "FoVx": focal2fov(fx, w),
            "FoVy": focal2fov(fy, h),
            "width": w,
            "height": h,
            "cx": cx,
            "cy": cy,
            "geom_vertices": verts,
            "geom_faces": self.mesh_topology["vert_ids"],
        }

        if "additional" in pkg:
            final["additional"] = pkg.additional

        if "A" in pkg and "W" in pkg:
            final["A"] = pkg.A
            final["W"] = pkg.W

        if "image_path" in pkg:
            final["image_path"] = pkg.image_path

        if "flame_path" in pkg:
            final["flame_path"] = pkg.flame_path

        return final
