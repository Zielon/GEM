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
import json
import os
from pathlib import Path
from PIL import Image
from loguru import logger
import cv2
import numpy as np
import torch as th
import trimesh
from data.base import BaseDataset, DatasetMode
from data.utils import load_obj, opengl_to_opencv
from utils.geometry import AttrDict
from data.mesh import to_mesh, to_canonical


def to_dict(npz):
    params = {}
    for key in ["neck_pose", "jaw_pose", "eyes_pose", "shape", "expr", "static_offset", "rotation", "translation"]:
        val = np.array(npz[key]).astype(np.float32)
        if key != "shape":
            val = val[0:1]
        else:
            val = val[None]
        params[key] = val
    return params


def get_transforms(path):
    if not Path(path).exists():
        raise ValueError(f"Path {path} not found!")
    f = open(path)
    data = json.load(f)
    f.close()
    return data


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def deep_merge(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1:
            if isinstance(dict1[key], list) and isinstance(value, list):
                dict1[key].extend(value)  # Concatenate lists
            elif isinstance(dict1[key], dict) and isinstance(value, dict):
                deep_merge(dict1[key], value)  # Recursive merge for dicts
            else:
                dict1[key] = value  # Override with new value
        else:
            dict1[key] = value  # Add new key


class NersembleDataset(BaseDataset):
    def __init__(self, config, custom_camera_list=None, mode=DatasetMode.train):
        super().__init__(config, custom_camera_list, mode)
        data = config.data
        self.imagepath = data.image
        self.maskpath = data.mask
        self.meshpath = data.mesh
        self.ds_rate = config.ds_rate
        self.config = config
        self.mode = mode
        self.mesh_topology = None
        self.custom_camera_list = custom_camera_list
        self.capture_id = config.capture_id
        self.root = str(Path(self.config.actor_id).parent)
        self.test_camera = config.data.test_camera

        texmean = np.asarray(Image.open(f"{str(Path(__file__).parent.parent)}/assets/textures/nersemble/flame_texture.png"), dtype=np.float32)
        texmean = np.copy(np.flip(texmean, 0))
        self.texmean = th.from_numpy(texmean).cuda().float().permute(2, 0, 1) / 255

        if mode == DatasetMode.train:
            self.transforms = get_transforms(config.actor_id + "/transforms_train.json")

        if mode == DatasetMode.validation:
            self.transforms = get_transforms(config.actor_id + "/transforms_val.json")

        if mode == DatasetMode.test:
            self.transforms = get_transforms(config.actor_id + "/transforms_test.json")

        join_configs = config.data.get("join_configs", False)
        if join_configs:
            self.transforms = self.join_configs(["transforms_train", "transforms_val"])

        self.create_cameras()
        self.create_topology()
        self.create_frames()
        self.set_identity()
        self.filter_frames()

    def set_identity(self):
        self.identity_frame = None
        identity_frame = self.config.data.identity_frame
        for frame in self.join_configs(["transforms_train", "transforms_val", "transforms_test"])["frames"]:
            frame_id = str(frame["timestep_index"]).zfill(5)
            cam_id = str(frame["camera_index"]).zfill(2)
            if f"{frame_id}_{cam_id}" == identity_frame:
                imagepath = self.root + "/" + frame["file_path"][2:]
                self.identity_frame = imagepath

        if self.identity_frame is None:
            logger.error(f"Idenity frame {identity_frame} was not found!")
            exit(-1)


    def join_configs(self, transforms):
        main = {}

        logger.warning(f"Joining configs: {transforms}")

        for name in transforms:
            file_path = os.path.join(self.config.actor_id, f"{name}.json")
            transforms_data = load_json(file_path)
            
            if not main:
                main = copy.deepcopy(transforms_data)  # Initialize with first config
            else:
                deep_merge(main, transforms_data)  # Merge subsequent configs

        if "frames" in main and isinstance(main["frames"], list):
            main["frames"].sort(key=lambda x: x.get("file_path", ""))

        return main

    def filter_frames(self):
        if self.custom_camera_list is None:
            return
        filtered = []
        for frame in self.frame_list:
            cam_idx = str(frame["camera_index"]).zfill(2)
            if cam_idx in self.custom_camera_list:
                filtered.append(frame)

        self.frame_list = filtered
        self.frame_list.sort(key=lambda x: (x.get("timestep_index", float("inf")), x.get("camera_index", float("inf"))))

    def create_cameras(self):
        self.allcameras = list(map(lambda e: str(e).zfill(2), range(16)))

    def create_frames(self):
        frames = []
        # for frame in sorted(self.transforms["frames"], key=lambda x: x["file_path"]):
        for frame in self.transforms["frames"]:
            frames.append(frame)
        self.frame_list = frames

    def create_topology(self):
        obj = load_obj(f"{Path(__file__).parent.parent}/assets/meshes/flame.obj")
        self.mesh_topology = obj

    def get_canonical_mesh(self):
        frame = self.frame_list[0]
        flame_params = np.load(self.parse(frame["flame_param_path"]))
        flame_params = to_dict(flame_params)
        mesh = to_canonical(flame_params)
        verts = mesh.vertices
        faces = mesh.faces

        logger.info(f"Loaded canonical mesh with {len(verts)} vertices and {len(faces)} faces")

        return th.from_numpy(verts).float().cuda(), th.from_numpy(faces).long().cuda()

    def parse(self, path):
        if Path(path).is_absolute():
            return path
        else:
            return self.root + "/" + path[2:]

    def get(self, idx):
        try:
            frame = self.frame_list[idx]

            cx = frame["cx"]
            cy = frame["cy"]
            fl_x = frame["fl_x"]
            fl_y = frame["fl_y"]
            h = frame["h"]
            w = frame["w"]

            sentnum = Path(frame["file_path"]).parent.parent.name
            timestep_index = frame["timestep_index"]
            cam = str(frame["camera_index"]).zfill(2)
            cam_id = frame["camera_index"]
            Rt = np.array(frame["transform_matrix"])

            Rt = opengl_to_opencv(Rt)

            K = np.eye(3)
            K[0, 2] = cx
            K[1, 2] = cy
            K[0, 0] = fl_x
            K[1, 1] = fl_y

            image_path = self.parse(frame["file_path"])
            alpha_path = image_path.replace("images", "alpha")
            flame_path = self.parse(frame["flame_param_path"])

            image = self.read_image(image_path, cv2.INTER_CUBIC) / 255
            alpha = self.read_image(alpha_path, cv2.INTER_CUBIC)[0:1, :, :] / 255
            flame_params = np.load(flame_path)
            flame_params = to_dict(flame_params)

            R = th.from_numpy(flame_params["rotation"]).float()
            T = th.from_numpy(flame_params["translation"]).float()

            mesh, J, A, W = to_mesh(flame_params, R=R, T=T)

            R = A[0, 0, :3, :3]
            T = A[0, 0, :3, 3] + T

            root_Rt = np.eye(4)
            root_Rt[:3, :3] = R.numpy()
            root_Rt[:3, 3] = T.numpy()

            mesh, J, A, W = to_mesh(flame_params)
            verts = mesh.vertices

            pkg = {
                "K": K,
                "Rt": Rt,
                "cam_id": cam_id,
                "image": image,
                "alpha": alpha,
                "verts": verts,
                "root_Rt": root_Rt,
                "frame": timestep_index,
                "sentnum": sentnum,
                "cam": cam,
                "flame_path": flame_path,
                "image_path": image_path,
                "A": A.numpy(),
                "W": W.numpy(),
            }

            return AttrDict(pkg)
        except Exception as e:
            logger.error(f"Error in get: {e}")
