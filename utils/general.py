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


import os
import random
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import importlib
import importlib.util
import copy
from loguru import logger
from enum import Enum
from torch.utils.data import DataLoader, Dataset
from typing import Any, Mapping, Tuple
from omegaconf import DictConfig, ListConfig, OmegaConf
from data.base import DatasetMode

from data.multiface import MultifaceDataset
from data.vhap import VHAPDataset
from data.nersamble import NersembleDataset
from data.transfer import TransferDataset
from data.audio import AudioDataset
from data.video import VideoDataset


def to_device(values, device=None, non_blocking=True):
    """Transfer a set of values to the device.
    Args:
        values: a nested dict/list/tuple of tensors
        device: argument to `to()` for the underlying vector
    NOTE:
        if the device is not specified, using `th.cuda()`
    """
    if device is None:
        device = th.device("cuda")

    if isinstance(values, dict):
        return {k: to_device(v, device=device) for k, v in values.items()}
    elif isinstance(values, tuple):
        return tuple(to_device(v, device=device) for v in values)
    elif isinstance(values, list):
        return [to_device(v, device=device) for v in values]
    elif isinstance(values, th.Tensor):
        return values.to(device, non_blocking=non_blocking)
    elif isinstance(values, nn.Module):
        return values.to(device)
    else:
        return values


def to_tensor(values, device="cuda"):
    if isinstance(values, dict):
        return {k: th.from_numpy(v).to(device) if type(v) is np.ndarray else v for k, v in values.items()}
    if isinstance(values, list):
        return [th.from_numpy(v).to(device) for v in values]
    if isinstance(values, np.ndarray):
        return th.from_numpy(values).to(device)
    else:
        return values


def get_single(batch, index):
    single = {}
    for key in batch.keys():
        if index >= len(batch[key]):
            return None
        single[key] = batch[key][index]
    return single


def copy_state_dict(cur_state_dict, pre_state_dict, prefix="", load_name=None):
    def _get_params(key):
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None

    for k in cur_state_dict.keys():
        if load_name is not None:
            if load_name not in k:
                continue
        v = _get_params(k)
        try:
            if v is None:
                # print('parameter {} not found'.format(k))
                continue
            cur_state_dict[k].copy_(v)
        except:
            # print('copy param {} failed'.format(k))
            continue


def seed_everything(seed=17):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)


def load_module(module_name, class_name=None, silent: bool = False):
    module = importlib.import_module(module_name)
    return getattr(module, class_name) if class_name else module


def load_class(class_name):
    return load_module(*class_name.rsplit(".", 1))


def instantiate(config, **kwargs):
    config = copy.deepcopy(config)
    class_name = config.pop("class_name")
    object_class = load_class(class_name)
    instance = object_class(**config, **kwargs)

    return instance


def seed_worker(worker_id):
    worker_seed = th.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


from torch.utils.data import Dataset, DataLoader, Sampler
import torch


class ConsecutiveSampler(Sampler):
    def __init__(self, data_source, cameras, batch_frames=4, views_per_frame=2, randomize_frames=False):
        """
        data_source: The dataset.
        cameras: Total number of views per frame.
        batch_frames: Number of consecutive frames per batch.
        views_per_frame: Number of views to sample per frame (must be <= cameras).
        randomize_frames: If True, the starting frame for each batch is chosen at random.
        """
        assert views_per_frame <= cameras, "views_per_frame must be <= cameras"
        
        self.data_source = data_source
        self.cameras = cameras
        self.batch_frames = batch_frames
        self.views_per_frame = views_per_frame
        self.randomize_frames = randomize_frames

        # Total number of frames available in the dataset.
        self.total_frames = len(self.data_source) // self.cameras

    def __iter__(self):
        if self.randomize_frames:
            # Choose every possible starting frame that allows a full batch.
            possible_starts = list(range(self.total_frames - self.batch_frames + 1))
            random.shuffle(possible_starts)
            for start in possible_starts:
                yield self._batch_indices(start)
        else:
            # Go through frames sequentially, non-overlapping batches.
            for start in range(0, self.total_frames - self.batch_frames + 1, self.batch_frames):
                yield self._batch_indices(start)

    def _batch_indices(self, start_frame):
        """Given a starting frame, return the indices for batch_frames consecutive frames,
        each with a random sample of views_per_frame camera views."""
        batch_indices = []
        for frame in range(start_frame, start_frame + self.batch_frames):
            frame_start_idx = frame * self.cameras
            # Create a list of indices for all camera views in the current frame.
            all_views = list(range(frame_start_idx, frame_start_idx + self.cameras))
            # Randomly sample the desired number of views from this frame.
            sampled_views = random.sample(all_views, self.views_per_frame)
            batch_indices.extend(sampled_views)
        return batch_indices

    def __len__(self):
        if self.randomize_frames:
            return self.total_frames - self.batch_frames + 1
        else:
            return (self.total_frames - self.batch_frames + 1) // self.batch_frames


def build_loader(dataset: Dataset, batch_size: int, num_workers: int = 0, shuffle: bool = True, persistent_workers: bool = True, camera_list = [], seed=33, use_sampler=False, **kwargs):
    generator = th.Generator()
    generator.manual_seed(seed)

    batch_sampler = None
    if batch_size > 1 and use_sampler:
        logger.warning("Using Consecutive Sampler")
        batch_sampler = ConsecutiveSampler(
            data_source=dataset,
            cameras=len(camera_list),
            batch_frames=1,
            views_per_frame=batch_size,
            randomize_frames=True
        )
        shuffle = False
        batch_size = 1

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=5,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=True,
        batch_sampler=batch_sampler,
    )


def build_dataset(config, camera_list=None, mode=DatasetMode.train, source_config=None):
    selected = config.get("dataset_name", None)
    if source_config != None:
        selected = "TRANSFER"
    if source_config != None and ".mp4" in source_config:
        selected = "VIDEO"

    camera_list = config.data.get("camera_list", None) if camera_list is None else camera_list

    logger.warning(f"Bulding {selected} dataset with {mode}")

    if selected == "MULTIFACE":
        return MultifaceDataset(config, camera_list, mode)
    if selected == "NERSEMBLE":
        return NersembleDataset(config, camera_list, mode)
    if selected == "TRANSFER":
        return TransferDataset(source_config, config, mode)
    if selected == "VHAP":
        return VHAPDataset(config, camera_list, mode)
    if selected == "VIDEO":
        return VideoDataset(config, source_config)

    raise NotImplementedError("Dataset not supported!")
