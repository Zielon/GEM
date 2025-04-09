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


import math

import torch
from torch import nn

from styleunet.modules import *
from utils.geometry import AttrDict


class Decoder(nn.Module):
    def __init__(self, output_dim, style_dim, channels, out_log_size, comb_num) -> None:
        super().__init__()
        self.comb_num = comb_num
        self.style_dim = style_dim
        self.channels = channels
        self.out_log_size = out_log_size
        self.convs = nn.ModuleList()
        self.to_ouptut = nn.ModuleList()

        in_channel = channels[8]
        n = out_log_size + 1 - 4 - 1

        for j, i in enumerate(range(4, out_log_size + 1)):
            out_channel = channels[2**i]
            enabled = True if j < n else False  # disable ativation for the last layer
            self.convs.append(StyledConv(in_channel, out_channel, 3, style_dim, upsample=True))
            self.convs.append(StyledConv(out_channel, out_channel, 3, style_dim, upsample=False, activation=enabled))
            self.to_ouptut.append(ToGaussian(out_channel, style_dim, output_dim))

            in_channel = out_channel

        self.iwt = InverseHaarTransform(3)

    def make_noise(self, device):
        noises = []
        for i in range(4, self.out_log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2**i, 2**i, device=device))
        return noises

    def forward(self, latent, cond_list, comb_convs):
        noises = self.make_noise(latent.device)
        i = 0
        skip = None
        for conv1, conv2, to_rgb in zip(self.convs[::2], self.convs[1::2], self.to_ouptut):
            if i == 0:
                out = comb_convs[self.comb_num](cond_list[self.comb_num])
            elif i < self.comb_num * 2 + 1:
                out = torch.cat([out, cond_list[self.comb_num - (i // 2)]], dim=1)
                out = comb_convs[self.comb_num - (i // 2)](out)

            enabled = 1.0 if i < len(self.to_ouptut) * 2 - 2 else 0.0

            out = conv1(out, latent, noises[i])
            out = conv2(out, latent, noises[i + 1] * enabled)
            skip = to_rgb(out, latent, skip)

            i += 2

        image = self.iwt(skip)

        return image


class DoubleGenerator(nn.Module):
    def __init__(self, map_channels, style_dim, channels, out_log_size, comb_num, output_dim, in_log_size, input_size, blur_kernel) -> None:
        super().__init__()

        self.zero_style = torch.ones([1, style_dim]).cuda().float() * 0.001

        self.encoder = Encoder(map_channels, blur_kernel, channels, input_size, in_log_size)
        self.decoder = Decoder(output_dim, style_dim, channels, out_log_size, comb_num)

    def forward(self, cond_image, latent, viewdir=None):
        cond_list = self.encoder(cond_image)

        if latent is None:
            B = cond_image.shape[0]
            latent = self.zero_style.expand(B, -1)

        pred = self.decoder(latent, cond_list, self.encoder.comb_convs)

        return pred


class Encoder(nn.Module):
    def __init__(self, map_channels, blur_kernel, channels, input_size, in_log_size) -> None:
        super().__init__()

        # add new layer here
        self.dwt = HaarTransform(3)
        self.from_rgbs = nn.ModuleList()
        self.cond_convs = nn.ModuleList()
        self.comb_convs = nn.ModuleList()

        in_channel = channels[input_size]
        for i in range(in_log_size - 2, 2, -1):
            out_channel = channels[2**i]
            self.from_rgbs.append(FromRGB(in_channel, map_channels, downsample=True))
            self.cond_convs.append(ConvBlock(in_channel, out_channel, blur_kernel))
            if i > 3:
                self.comb_convs.append(ConvLayer(out_channel * 2, out_channel, 3))
            else:
                self.comb_convs.append(ConvLayer(out_channel, out_channel, 3))
            in_channel = out_channel

    def forward(self, cond):
        cond_img = self.dwt(cond)
        cond_out = None
        cond_list = []
        for from_rgb, cond_conv in zip(self.from_rgbs, self.cond_convs):
            cond_img, cond_out = from_rgb(cond_img, cond_out)
            cond_out = cond_conv(cond_out)
            cond_list.append(cond_out)

        return cond_list


class StyleUNet(nn.Module):
    def __init__(self, map_channels, input_size, output_size, style_dim=64, mlp_num=4, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.style_dim = style_dim
        self.mlp_num = mlp_num

        self.pixelnorm = PixelNorm()
        mlp = []
        for i in range(mlp_num):
            mlp.append(nn.Linear(style_dim, style_dim, bias=True))
            if i < mlp_num - 1:
                mlp.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        self.mapping = nn.Sequential(*mlp)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.in_log_size = int(math.log(input_size, 2)) - 1
        self.out_log_size = int(math.log(output_size, 2)) - 1
        self.comb_num = self.in_log_size - 5

        self.color = TripletGenerator(
            map_channels, style_dim, self.channels, self.out_log_size, self.comb_num, 3, self.in_log_size, input_size, blur_kernel
        )
        self.position = TripletGenerator(
            map_channels, style_dim, self.channels, self.out_log_size, self.comb_num, 3, self.in_log_size, input_size, blur_kernel
        )
        self.gaussians = TripletGenerator(
            map_channels, style_dim, self.channels, self.out_log_size, self.comb_num, 8, self.in_log_size, input_size, blur_kernel
        )

    def forward(self, condition_img, style, viewdir):
        latent = self.mapping(self.pixelnorm(style))

        front_color, back_color = self.color(condition_img, latent, viewdir)
        front_position, back_position = self.position(condition_img, latent)
        front_gaussians, back_gaussians = self.gaussians(condition_img, latent)

        return (front_color, front_position, front_gaussians), (back_color, back_position, back_gaussians)


class StyleUNetLight(nn.Module):
    def __init__(self, config, map_channels, input_size, output_size, style_dim=64, mlp_num=4, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.style_dim = style_dim
        self.mlp_num = mlp_num
        self.config = config
        self.fix_rgb = config.train.get("fix_rgb", True)
        self.use_sh = config.train.get("use_sh", False)
        self.use_shadow = config.train.get("use_shadow", False)
        self.use_view_cond = config.train.get("use_view_cond", False)
        self.use_style = config.train.get("use_style", True)
        self.use_feature_map = config.train.get("use_feature_map", True)

        self.pixelnorm = PixelNorm()
        mlp = []
        for i in range(mlp_num):
            mlp.append(nn.Linear(self.style_dim, self.style_dim, bias=True))
            if i < mlp_num - 1:
                mlp.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        self.mapping = nn.Sequential(*mlp)

        unet_scale = config.train.get("unet_scale", 1)

        self.channels = {
            4: 256 * unet_scale,
            8: 256 * unet_scale,
            16: 256 * unet_scale,
            32: 256 * unet_scale,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.in_log_size = int(math.log(input_size, 2)) - 1
        self.out_log_size = int(math.log(output_size, 2)) - 1
        self.comb_num = self.in_log_size - 5

        color_output = 0

        if self.use_sh:
            color_output = 45

        if self.use_shadow:
            color_output = 1

        self.apperance = DoubleGenerator(
            map_channels, style_dim, self.channels, self.out_log_size, self.comb_num, 3, self.in_log_size, input_size, blur_kernel
        )

        # Use only one geomtery
        if self.use_feature_map:
            feat_dim = 32
            self.feature_map = nn.Parameter(torch.randn(1, feat_dim, input_size, input_size) * 0.1)
            map_channels += feat_dim

        self.geometry = DoubleGenerator(
            map_channels,
            style_dim,
            self.channels,
            self.out_log_size,
            self.comb_num,
            3 + 1 + 3 + 4 + color_output,
            self.in_log_size,
            input_size,
            blur_kernel,
        )

    def forward(self, condition_img, canon_img, style, view_cond=None):
        latent = None
        if self.use_style:
            latent = self.mapping(self.pixelnorm(style))

        if self.fix_rgb:
            # cond = canon_img
            # if self.use_view_cond and view_cond is not None:
            #     cond = torch.cat([canon_img, view_cond], dim=1)
            app = self.apperance(canon_img, latent, None)
        else:
            app = self.apperance(condition_img, latent, None)

        if self.use_feature_map:
            condition_img = torch.cat([condition_img, self.feature_map.expand(condition_img.shape[0], -1, -1, -1)], dim=1)

        geom = self.geometry(condition_img, latent)

        output = {
            # Dynamic part
            "scales": geom[:, 0:3, :, :],
            "rotation": geom[:, 3:7, :, :],
            "position": geom[:, 7:10, :, :],
            "opacity": geom[:, 10:11, :, :],
            # Static part
            "rgb": app[:, 0:3, :, :],
        }

        # Dynamic part
        if self.use_shadow:
            output["shadow"] = geom[:, 11:12, :, :]

        if self.use_sh:
            output["shs_fc"] = geom[:, 11:56, :, :]

        return AttrDict(output)
