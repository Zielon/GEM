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
import torch.nn as nn
import torch as th
from gaussians.utils import inverse_sigmoid

VOL_TRACKER_PATH = os.environ["VOL_TRACKER_PATH"]


class BaseModel:
    def __init__(self, config, dataset) -> None:
        self.dataset = dataset
        self.config = config

        self.uv_size = config.train.get("uv_size", 128)
        self.bg_color = config.train.get("bg_color", "black")
        self.pca_n_components = config.train.get("pca_n_components", 16)
        self.enable_bg_from = config.train.get("enable_bg_from", False)
        self.use_pca_layer = config.train.get("use_pca_layer", True)
        self.use_def_grad_map = config.train.get("use_def_grad_map", False)
        self.use_uv_sampling = config.train.get("use_uv_sampling", False)
        self.use_bg_net = config.train.get("use_bg_net", False)
        self.use_color_calib = config.train.get("use_color_calib", False)
        self.use_pixel_bias = config.train.get("use_pixel_bias", False)
        self.use_texture_codes = config.train.get("use_texture_codes", False)
        self.use_sh = config.train.get("use_sh", False)
        self.use_shadow = config.train.get("use_shadow", False)
        self.use_sploc = config.train.get("use_sploc", False)
        self.use_view_cond = config.train.get("use_view_cond", False)
        self.use_style = config.train.get("use_style", True)
        self.weight_scale_reg = config.train.get("weight_scale_reg", 20.0)
        self.weight_pos_reg = config.train.get("weight_pos_reg", 750.0)
        self.weight_shadow_reg = config.train.get("weight_shadow_reg", 0.01)

        self.curr_iter = 0

        if self.use_sh and self.use_shadow:
            raise RuntimeError("You cannot use SH and Shadow")

        self.scaling_activation = th.exp
        self.scaling_inverse_activation = th.log
        self.opacity_activation = th.sigmoid
        self.color_activation = th.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = th.nn.functional.normalize

    def disable_grad(self):
        raise NotImplementedError()

    def count_parameters(self):
        raise NotImplementedError()

    def create(self):
        raise NotImplementedError()

    def get_opt_params(self):
        raise NotImplementedError()

    def predict(self, batch):
        raise NotImplementedError()

    def state_dict(self):
        raise NotImplementedError()

    def load_state_dict(self):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()