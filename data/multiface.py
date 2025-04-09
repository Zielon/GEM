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
import math
import os
from pathlib import Path
import sys
import cv2

from random import randint
from loguru import logger
import numpy as np
import torch as th
from PIL import Image
from tqdm import tqdm
from data.base import BaseDataset, DatasetMode
from data.utils import load_obj
from utils.geometry import AttrDict

VOL_TRACKER_PATH = os.environ["VOL_TRACKER_PATH"]


def check_path(path):
    if not os.path.exists(path):
        sys.stderr.write("%s does not exist!\n" % (path))
        sys.exit(-1)


def load_krt(path):
    cameras = {}

    with open(path, "r") as f:
        while True:
            name = f.readline()
            if name == "":
                break

            intrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            dist = [float(x) for x in f.readline().split()]
            extrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            f.readline()

            cameras[name[:-1]] = {
                "K": np.array(intrin),
                "Rt": np.array(extrin),
            }

    return cameras


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


class MultifaceDataset(BaseDataset):
    def __init__(self, config, custom_camera_list=None, mode=DatasetMode.train):
        super().__init__(config, custom_camera_list, mode)
        data = config.data
        self.camera_ids = {}
        self.cache = {}
        self.capture_id = config.capture_id
        self.exclude = config.data.exclude
        self.exclude_cameras = config.data.exclude_cameras
        self.gamma_space = True

        frame_list = np.genfromtxt(data.frame_list, dtype=(str, str))
        if mode != DatasetMode.test:
            frame_list = self.crop_sentences(frame_list)

        # set cameras
        krt = load_krt(data.krt_dir)
        self.krt = krt
        self.cameras = list(krt.keys())
        for i, k in enumerate(self.cameras):
            self.camera_ids[k] = i
        self.allcameras = sorted(self.cameras)
        self.vertmean = np.fromfile("{}/vert_mean.bin".format(config.actor_id), dtype=np.float32)
        self.vertstd = float(np.genfromtxt("{}/vert_var.txt".format(config.actor_id)) ** 0.5)

        texmean = np.asarray(Image.open("{}/tex_mean.png".format(config.actor_id)), dtype=np.float32)
        texmean = np.copy(np.flip(texmean, 0))
        self.texmean = th.from_numpy(texmean).cuda().float().permute(2, 0, 1) / 255
        self.extra_topology = th.load(f"{VOL_TRACKER_PATH}/assets/multiface_topology.ptk")

        # set frames
        self.initialize()
        self.parse_frames(frame_list)
        self.filter_frames()
        self.check_available_frames()

        self.frame_to_id = {}
        i = 0
        for frame in self.frame_list:
            sentnum, frame, cam  = frame
            if frame not in self.frame_to_id:
                self.frame_to_id[frame] = i
                i += 1

    def crop_sentences(self, frame_list):
        curr_sentnum = tuple(frame_list[0])[0]
        ranges = []
        j = 0
        for i, line in enumerate(frame_list):
            sentnum, frame = tuple(line)
            seq_parts_set = set(sentnum.split("_"))
            exclude_parts_set = set(self.exclude)
            if len(seq_parts_set.intersection(exclude_parts_set)) > 0:
                continue
            if sentnum == curr_sentnum:
                j += 1
                continue
            ranges.append((i - j, i, curr_sentnum))
            curr_sentnum = sentnum
            j = 0

        blocks = []
        for start, end, sentnum in ranges:
            trim = int((end - start) * 0.2)  # trim 20% of start and end
            start = min(start + trim, end)
            end = max(start, end - trim)

            block = frame_list[start:end]
            blocks.append(block)

        return np.concatenate(blocks, axis=0)

    def initialize(self):
        """
        vt: [n_uv_coords, 2] th.Tensor
        UV coordinates.
        vi: [..., 3] th.Tensor
        Face vertex indices.
        vti: [..., 3] th.Tensor
        Face UV indices.
        """
        path = self.config.data.canonical_mesh
        obj = load_obj(path.replace("bin", "obj"))
        self.mesh_topology = obj

    def get_canonical_mesh(self):
        verts = np.fromfile(self.config.data.canonical_mesh, dtype=np.float32)
        verts *= 0.001
        verts = verts.reshape((-1, 3)).astype(np.float32)
        faces = self.mesh_topology["vert_ids"]

        logger.info(f"Loaded canonical mesh with {len(verts)} vertices and {len(faces)} faces")

        return th.from_numpy(verts).float().cuda(), th.from_numpy(faces).long().cuda()

    def check_available_frames(self):
        available_frames = []
        available_sentences = set(map(lambda p: Path(p).stem, sorted(glob(self.imagepath + "/*"))))
        for i, x in tqdm(enumerate(self.frame_list)):
            sentnum, frame, cam = x
            if sentnum in available_sentences:
                available_frames.append(x)

        self.frame_list = available_frames

    def parse_frames(self, frame_list):
        frames = []
        for i, x in tqdm(enumerate(frame_list)):
            for i, cam in enumerate(self.cameras):
                f = tuple(x) + (cam,)
                s, f, c = f
                frames.append((str(s), f, c))

        self.frame_list = frames

    def filter_frames(self):
        filterd_frames = []
        filters = list(self.config.data.test_sequences)
        for i, x in enumerate(self.frame_list):
            sentnum = x[0]
            cond = sentnum in filters if not (self.mode == DatasetMode.test) else sentnum not in filters
            if cond:
                continue
            filterd_frames.append(x)

        cam_ids = set(self.cameras)
        cam_to_remove = set(self.exclude_cameras)
        ids = list(cam_ids - cam_to_remove)

        self.frame_list = self.filter_cameras(filterd_frames, cams=ids)

        if self.custom_camera_list is not None:
            self.frame_list = self.filter_cameras(self.frame_list, cams=self.custom_camera_list)

    def filter_cameras(self, frames, cams):
        output = []
        for frame in frames:
            _, _, cam = frame
            if cam in cams:
                output.append(frame)
        return output

    def get(self, idx):
        try:
            frame_pkg = self.frame_list[idx]
            while self.until_valid(frame_pkg):
                idx = randint(0, len(self.frame_list) - 1)
                frame_pkg = self.frame_list[idx]

            sentnum, frame, cam = frame_pkg
            cam_id = self.camera_ids[cam]

            sentnum, frame, cam = self.frame_list[idx]
            path = "{}/{}/{}/{}.png".format(self.imagepath, sentnum, cam, frame)
            image = self.read_image(path, cv2.INTER_CUBIC) / 255

            sentnum, frame, cam = self.frame_list[idx]
            path = "{}/{}/{}/{}.png".format(self.maskpath, sentnum, cam, frame)
            alpha = self.read_image(path, cv2.INTER_NEAREST)[0:1, :, :] / 255

            # camera
            root_Rt = np.genfromtxt("{}/{}/{}_transform.txt".format(self.meshpath, sentnum, frame)).astype(np.float32)
            root_Rt[:3, 3] *= 0.001

            Rt, K = self.krt[cam]["Rt"].copy().astype(np.float32), self.krt[cam]["K"].copy().astype(np.float32)
            Rt[:3, 3] *= 0.001

            path = "{}/{}/{}.bin".format(self.meshpath, sentnum, frame)
            verts = np.fromfile(path, dtype=np.float32)
            verts *= 0.001
            verts = verts.reshape((-1, 3)).astype(np.float32)

            return AttrDict(
                {
                    "K": K,
                    "Rt": Rt,
                    "cam_id": cam_id,
                    "image": image,
                    "alpha": alpha,
                    "verts": verts,
                    "root_Rt": root_Rt,
                    "frame": frame,
                    "sentnum": sentnum,
                    "cam": cam,
                }
            )
        except Exception as e:
            logger.error(f"Error in get: {e}")