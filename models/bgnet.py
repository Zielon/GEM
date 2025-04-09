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


import torch
import torch.nn as nn

from typing import Tuple, List, Union

from models.embeddings import Optcodes, PositionalEncoding


class BgNet(nn.Module):
    def __init__(
        self,
        input_size: int = 3,
        n_dims: int = 128,
        n_views: int = 4,
        W: int = 128,
        img_res: Union[Tuple, List] = (1024, 667),
    ):
        super().__init__()
        self.optcodes = Optcodes(n_views, n_dims)
        self.sigmoid = torch.nn.Sigmoid()
        self.posi_enc = PositionalEncoding(2, num_freqs=7)
        self.network = nn.Sequential(
            nn.Linear(n_dims + self.posi_enc.dims + input_size, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
            nn.Linear(W, 3),
        )
        self.img_res = img_res

        nn.init.normal_(self.network[-1].weight.data, 0, 0.001)
        if self.network[-1].bias is not None:
            nn.init.constant_(self.network[-1].bias.data, 0.0)

        img_h, img_w = self.img_res

        w, h = torch.meshgrid(
            torch.arange(img_w, device="cuda"),
            torch.arange(img_h, device="cuda"),
            indexing="xy",
        )
        w = w.flatten()
        h = h.flatten()

        pixel_h = h.float() / (img_h - 1) * 2.0 - 1.0
        pixel_w = w.float() / (img_w - 1) * 2.0 - 1.0

        self.pixel_locs = self.posi_enc(torch.stack([pixel_h, pixel_w], dim=-1))[0]

    def forward(self, batch):
        fg = batch["alpha"]
        bg = (1 - fg) * batch["image"]
        cam_idxs = batch["cam"][None]
        H, W = self.img_res

        framecode = self.optcodes(cam_idxs).expand(self.pixel_locs.shape[0], -1)
        input_feat = torch.cat([self.pixel_locs, bg.reshape(-1, 3), framecode], dim=-1)

        bg_preds = self.network(input_feat).reshape(3, H, W)

        return bg_preds
