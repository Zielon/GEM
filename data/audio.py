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


class AudioDataset(BaseDataset):
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
        for frame in self.join_configs(["transforms_train", "transforms_test"])["frames"]:
            frame_id = str(frame["timestep_index"]).zfill(5)
            cam_id = str(frame["camera_index"]).zfill(2)
            if f"{frame_id}_{cam_id}" == identity_frame:
                imagepath = self.root + "/" + frame["file_path"][2:]
                self.identity_frame = imagepath

        if self.identity_frame is None:
            logger.error(f"Idenity frame {identity_frame} was not found!")
            exit(-1)

    def join_configs(self, configs):
        main = None
        for name in configs:
            data = get_transforms(self.config.actor_id + f"/{name}.json")
            if main is None:
                main = copy.deepcopy(data)
            else:
                main["frames"] += data["frames"]
                main["camera_indices"] += data["camera_indices"]
                main["timestep_indices"] += data["timestep_indices"]

        main["frames"] = sorted(main["frames"], key=lambda x: x["file_path"])
        main["camera_indices"] = sorted(main["camera_indices"])
        main["timestep_indices"] = sorted(main["timestep_indices"])

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

        K = np.eye(3)
        K[0, 2] = cx
        K[1, 2] = cy
        K[0, 0] = fl_x
        K[1, 1] = fl_y

        image_path = self.parse(frame["file_path"])
        alpha_path = self.parse(frame["fg_mask_path"])
        flame_path = self.parse(frame["flame_param_path"])

        image = self.read_image(image_path, cv2.INTER_CUBIC) / 255
        alpha = self.read_image(alpha_path, cv2.INTER_CUBIC)[0:1, :, :] / 255

        root_Rt = np.eye(4)
        mesh = trimesh.load(flame_path.replace(".npz", ".ply"), process=False)

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
            "image_path": image_path
        }

        if self.include_lbs:
            lbs = th.load(flame_path.replace("flame_param", "lbs").replace(".npz", ".pt"))
            pkg["A"] = lbs["A"]
            pkg["W"] = lbs["W"]

        return AttrDict(pkg)
