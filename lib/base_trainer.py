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
from glob import glob
import os
from loguru import logger
import torch as th
import torch.nn.functional as F
from tqdm import tqdm
from utils.renderer import Renderer


class BaseTrainer:
    def __init__(self, config, dataset) -> None:
        self.config = config
        self.dataset = dataset
        self.max_iter = config.train.iterations
        self.max_iter += config.train.get("warmup", 0)
        self.curr_iter = 1
        self.run_dir = config.train.run_dir
        self.model = None
        self.progress_bar = None
        self.health_checkpoint = None
        self.tb_writer = None
        self.optimizer = None
        self.scheduler = None
        self.bg = None
        self.uv_size = config.train.get("uv_size", 128)
        self.bg_color = config.train.get("bg_color", "black")
        self.use_alpha_loss = config.train.get("use_alpha_loss", False)
        self.enable_vgg_from = config.train.get("enable_vgg_from", 300_000)
        self.renderer = Renderer(white_background=self.bg_color == "white").cuda()
        self.renderer.resize(self.config.height, self.config.width)

        self.initialize()
        self.print()

    def render_mesh(self, Rt, cameras, mesh, mask=None, bg_color=None):
        with th.no_grad():
            R = Rt[:3, :3]
            T = Rt[:3, 3]
            vertices = (R @ mesh.v.float().T).T + T
            faces = mesh.f.long()[None]
            mesh_rendering = self.renderer(cameras, vertices[None], faces).permute(2, 0, 1)
            if mask is not None:
                alpha = self.renderer.resterize_attributes(cameras, vertices[None], faces, mask)[0][0, :, :, 0].permute(2, 0, 1)
                if bg_color is None:
                    bg = th.zeros_like(mesh_rendering) if self.bg_color == "black" else th.ones_like(mesh_rendering)
                else:
                    bg = th.ones_like(mesh_rendering) * bg_color
                mesh_rendering = mesh_rendering * alpha + (1 - alpha) * bg
            return mesh_rendering

    def stringify(self, losses):
        strings = {}
        for k in losses.keys():
            loss = losses[k]
            if th.is_tensor(loss):
                loss = loss.item()
            if isinstance(loss, float):
                strings[k] = f"{loss:.{5}f}"
            else:
                strings[k] = loss
        return strings

    def log(self, info):
        n_log = self.config.train.log_n_steps
        if self.curr_iter % n_log == 0:
            for key in info.keys():
                self.tb_writer.add_scalar(f"loss/{key}", info[key], self.curr_iter)
            capture_id = self.config.capture_id
            self.progress_bar.set_description(capture_id)
            self.progress_bar.set_postfix(self.stringify(info))
            self.progress_bar.update(n_log)

    def save(self, iteration=-1, name=None):
        if name is not None:
            path = name
        else:
            path = f"/checkpoints/chkpnt" + str(iteration).zfill(6) + ".pth"

        path = self.run_dir + path
        model_params = self.state_dict()

        if iteration % 500 == 0:
            self.health_checkpoint = (copy.deepcopy(model_params), iteration)

        if (iteration % self.config.train.checkpoint_n_steps != 0) and name is None:
            return

        th.save((model_params, iteration), path)

        logger.info(f"\n[ITER {iteration}] Saving Checkpoint to {path}")

    def recover(self):
        model_params, iteration = self.health_checkpoint
        logger.warning(f"Loss became NaN. Recovering mode to last health checkpoint from {iteration}")
        self.model.load_state_dict(model_params)

    def load_state_dict(self, state):
        opt_params, model_params = state

        opt_dict, scheduler_dict = opt_params
        self.optimizer.load_state_dict(opt_dict)
        self.scheduler.load_state_dict(scheduler_dict)
        self.model.load_state_dict(model_params)

    def legacy_load_state_dict(self, state):
        model_dict, opt_dict, scheuler_dict, colorcal_dict, pixelcal_dict, bg_dict = state

        self.optimizer.load_state_dict(opt_dict)
        self.scheduler.load_state_dict(scheuler_dict)
        self.model.load_state_dict((model_dict, colorcal_dict, pixelcal_dict, bg_dict))

    def state_dict(self):
        opt_params = (
            self.optimizer.state_dict(),
            self.scheduler.state_dict(),
        )

        return (opt_params, self.model.state_dict())

    def restore(self, iteration=None, force=False):
        path = os.path.join(self.run_dir, "model.pth")
        if force:
            path = ""
        if not os.path.exists(path):
            path = os.path.join(self.run_dir, "checkpoints")
            checkpoints = sorted(glob(path + "/*.pth"))
            if len(checkpoints) == 0:
                return 1
            path = checkpoints[-1]
            if iteration is not None:
                for checkpoint in checkpoints:
                    if iteration in checkpoint:
                        path = checkpoint
                        break

        (model_params, first_iter) = th.load(path, weights_only=False)

        first_iter += 1

        if len(model_params) == 2:
            self.load_state_dict(model_params)
            logger.info(f"Initialized from {first_iter}th step {path}!")
        else:
            self.legacy_load_state_dict(model_params)
            logger.info(f"[LEGACY] Initialized from {first_iter}th step {path}!")

        self.curr_iter = first_iter

        return first_iter

    def step(self, batch):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def get_loss(self, batch):
        raise NotImplementedError()

    def inference(self, batch):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()

    def initialize(self):
        raise NotImplementedError()

    def print(self):
        raise NotImplementedError()