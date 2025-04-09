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
from data.base import BaseDataset, DatasetMode
from data.nersamble import NersembleDataset
from lib.F3DMM.FLAME2023.flame import FLAME
from utils.geometry import AttrDict
from loguru import logger
import cv2


def to_flame(npz):
    params = {}
    for key in ["neck_pose", "jaw_pose", "eyes_pose", "shape", "expr", "static_offset", "rotation", "translation"]:
        params[key] = npz[key]
    return AttrDict(params)


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()  # Read a frame
        if not ret:
            break  # Exit loop if no frame is returned
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC) / 255
        frame = np.transpose(frame[..., ::-1].astype(np.float32), axes=(2, 0, 1))
        frames.append(frame)
    
    cap.release()  # Release the video capture object
    return frames


### Video Dataset class
# NOTE that here we assumed that the video is already preprocessed using DECA cropping!
class VideoDataset(NersembleDataset):
    def __init__(self, config, video_path):
        super().__init__(config, None, DatasetMode.test)

        self.include_lbs = True
        self.flame = FLAME()
        self.source = NersembleDataset(config, ["08"], DatasetMode.test)
        i = self.find_frame_index(config.data.get("identity_frame", 0), self.source.frame_list)
        self.identity = self.source.get(i)

        self.source_netural = None
        path = self.identity.flame_path
        if os.path.exists(path):
            self.source_netural = to_flame(np.load(path))

        self.frame_list = extract_frames(video_path)

    def find_frame_index(self, frame_name, frame_list):
        for i, frame in enumerate(frame_list):
            frame_id = Path(frame["file_path"]).stem
            if frame_id == frame_name:
                return i

        raise RuntimeError(f"Identity frame was not found for {self.config.capture_id}")

    def get(self, idx):
        try:
            src_pkg = self.source.get(0)

            src_pkg.additional = self.frame_list[idx]

            return src_pkg
        except Exception as e:
            logger.error(f"Error in get: {e}")