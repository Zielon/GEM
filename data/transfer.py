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


import copy
import os
from pathlib import Path
import numpy as np
import torch as th
from scipy.spatial.transform import Rotation as R
from data.base import DatasetMode
from data.nersamble import NersembleDataset
from lib.F3DMM.FLAME2023.flame import FLAME
from utils.geometry import AttrDict
from loguru import logger


def to_flame(npz):
    params = {}
    for key in ["neck_pose", "jaw_pose", "eyes_pose", "shape", "expr", "static_offset", "rotation", "translation"]:
        params[key] = npz[key]
    return AttrDict(params)


class TransferDataset(NersembleDataset):
    def __init__(self, source_config, target_config, mode=DatasetMode.validation):
        super().__init__(target_config, None, mode)

        self.include_lbs = True

        if target_config.dataset_name.upper() != "NERSEMBLE":
            raise NotImplementedError("Currently only FLAME based models are supported!")

        self.flame = FLAME()

        selected = source_config.get("dataset_name", None)
        if selected == "NERSEMBLE":
            source_config.data.join_configs = False
            self.source = NersembleDataset(source_config, None, mode)

        i = self.find_frame_index(source_config.data.get("identity_frame", 0), self.source.frame_list)
        self.identity = self.source.get(i)

        self.source_netural = None
        path = self.identity.flame_path
        if os.path.exists(path):
            self.source_netural = to_flame(np.load(path))

        i = self.find_frame_index(target_config.data.get("identity_frame", 0), self.frame_list)
        self.target_identity = super().get(i)
        self.target_netural = to_flame(np.load(self.target_identity.flame_path))

        # Limit target to the max length of source dataset
        N = min(len(self.source.frame_list), len(self.frame_list))
        self.frame_list = self.frame_list[:N]

    def find_frame_index(self, frame_name, frame_list):
        for i, frame in enumerate(frame_list):
            frame_id = Path(frame["file_path"]).stem
            if frame_id == frame_name:
                return i

        raise RuntimeError(f"Identity frame was not found for {self.config.capture_id}")

    def morph(self, params):
        neck_pose = th.from_numpy(params.neck_pose).float()
        jaw_pose = th.from_numpy(params.jaw_pose).float()
        eyes_pose = th.from_numpy(params.eyes_pose).float()
        shape = th.from_numpy(params.shape)[None].float()
        expr = th.from_numpy(params.expr).float()
        R = th.zeros([1, 3])
        pose = th.cat([R, jaw_pose], dim=-1).float()

        static_offset = th.from_numpy(params["static_offset"])[:, :5023, :].float()

        vertices, J, A, W = self.flame(
            shape_params=shape, expression_params=expr, pose_params=pose, neck_pose=neck_pose, eye_pose=eyes_pose, delta=static_offset, transl=None
        )

        return vertices[0]

    def relative_rotation(self, src, target):
        r_src = R.from_rotvec(src).as_matrix()
        r_target = R.from_rotvec(target).as_matrix()
        return R.from_matrix(r_src @ np.linalg.inv(r_target)).as_rotvec()

    def transfer(self, src_params, target_params):
        # EXPRESSIONS
        relative_expr = src_params.expr - self.source_netural.expr
        target_params.expr = self.target_netural.expr + relative_expr
        # JAW
        relative_jaw = src_params.jaw_pose - self.source_netural.jaw_pose
        target_params.jaw_pose = self.target_netural.jaw_pose + relative_jaw
        # NECK
        relative_neck = src_params.neck_pose - self.source_netural.neck_pose
        target_params.neck_pose = self.target_netural.neck_pose + relative_neck
        # EYES
        relative_eyes = src_params.eyes_pose - self.source_netural.eyes_pose
        target_params.eyes_pose = self.target_netural.eyes_pose + relative_eyes

        # Get FLAME
        return self.morph(target_params), copy.deepcopy(target_params)

    def get(self, idx):
        try:
            target_pkg = super().get(idx)
            src_pkg = self.source.get(idx)

            new_params = {}
            if os.path.exists(src_pkg.flame_path):
                src_params = to_flame(np.load(src_pkg.flame_path))
                target_params = to_flame(np.load(target_pkg.flame_path))
                verts, new_params = self.transfer(src_params, target_params)
            else:
                verts = src_pkg.verts

            target_pkg.verts = verts
            target_pkg.image = src_pkg.image
            target_pkg.alpha = src_pkg.alpha
            target_pkg.root_Rt = src_pkg.root_Rt
            target_pkg.K = src_pkg.K
            target_pkg.Rt = src_pkg.Rt

            target_pkg.A = self.target_identity.A
            target_pkg.W = self.target_identity.W
            target_pkg.flame = new_params

            return target_pkg
        except Exception as e:
            logger.error(f"Error in get: {e}")
