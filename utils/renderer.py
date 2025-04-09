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
import numpy as np
import pytorch3d
import torch as th
import torch.nn.functional as F
import torch.nn as nn
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardFlatShader,
    TexturesVertex,
    PointLights,
    PerspectiveCameras,
    OrthographicCameras,
    BlendParams,
    look_at_view_transform,
)

eps: float = 1e-8


class Renderer(nn.Module):
    def __init__(self, white_background=True):
        super().__init__()

        raster_settings = RasterizationSettings(blur_radius=0.0, faces_per_pixel=1, perspective_correct=True, cull_backfaces=True)

        self.lights = PointLights(
            device="cuda:0",
            location=((0, 0, 1),),
            ambient_color=((0.45, 0.45, 0.45),),
            diffuse_color=((0.35, 0.35, 0.35),),
            specular_color=((0.05, 0.05, 0.05),),
        )

        bg_color = [1, 1, 1] if white_background else [0, 0, 0]

        self.blend = BlendParams(background_color=bg_color)
        self.rasterizer = MeshRasterizer(raster_settings=raster_settings)
        self.renderer = MeshRenderer(
            self.rasterizer,
            shader=HardFlatShader(device="cuda:0", lights=self.lights, blend_params=self.blend),
        )

        self.setup_cameras()

    def set_bg_color(self, color):
        self.blend = BlendParams(background_color=color)
        self.renderer.shader = HardFlatShader(device="cuda:0", lights=self.lights, blend_params=self.blend)

    def setup_cameras(self):
        R, T = look_at_view_transform(1.5, 5.0, 0.0)
        self.front_camera = OrthographicCameras(R=R, T=T, image_size=((512, 512),), focal_length=5).cuda()

        R, T = look_at_view_transform(1.5, 5.0, 180.0)
        self.back_camera = OrthographicCameras(R=R, T=T, image_size=((512, 512),), focal_length=5).cuda()

    @staticmethod
    def to_cameras(f) -> PerspectiveCameras:
        w2c, K, crop = (f["cam_RT"], f["K"], f["crop"])

        H = int(f["crop"][-1])
        W = int(f["crop"][-2])

        Rt = w2c[None].cuda().float()

        R = Rt[:, :3, :3]
        tvec = Rt[:, :3, 3]

        image_size = th.tensor([[H, W]]).cuda().int()

        cameras = cameras_from_opencv_projection(R, tvec, K[None], image_size)

        return cameras

    def resize(self, H, W):
        self.renderer.rasterizer.raster_settings.image_size = (H, W)
        self.rasterizer.raster_settings.image_size = (H, W)

    def forward(self, cameras, vertices, faces, verts_rgb=None, meshes=None):
        if meshes is None:
            B, N, V = vertices.shape
            if verts_rgb is None:
                verts_rgb = th.ones(1, N, V)
            textures = TexturesVertex(verts_features=verts_rgb.cuda())
            meshes = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)

        P = cameras.get_world_to_view_transform().inverse().get_matrix().transpose(1, 2)[:, :3, 3]
        self.renderer.shader.lights.location = P

        rendering = self.renderer(meshes, cameras=cameras)
        return rendering[0, :, :, 0:3]

    def resterize_attributes(self, cameras, vertices, faces, attributes):
        meshes = pytorch3d.structures.Meshes(verts=vertices, faces=faces)

        fragments = self.rasterizer(meshes, cameras=cameras)

        mask = (fragments.pix_to_face > 0).float()[:, :, :, 0:1]  # [n, y, x, k]

        resterizerd_attribs = interpolate_face_attributes(
            fragments.pix_to_face,
            fragments.bary_coords,
            attributes,
        )

        return resterizerd_attribs, mask

    def map(self, cameras, verts, faces, attributes, inflate=0):
        meshes = pytorch3d.structures.Meshes(verts=verts, faces=faces)
        # postion = deformed[0]  # meshes.verts_packed()  # (V, 3)
        N = faces.shape[1]
        H, W = self.rasterizer.raster_settings.image_size

        if inflate > 0:
            vertex_normals = meshes.verts_normals_packed()  # (V, 3)
            verts = meshes.verts_packed()
            inflated_verts = verts + inflate * vertex_normals
            meshes = pytorch3d.structures.Meshes(verts=[inflated_verts], faces=meshes.faces_list())
    
        # faces_postions = postion[faces][0]
        # faces_deformations = def_grad[:, None].expand(-1, 3, -1, -1).reshape(N, 3, -1)
        # faces_postions_view = cameras.get_world_to_view_transform().transform_points(postion)[faces][0]
        # 
        # faces_normals = vertex_normals[faces][0]

        fragments = self.rasterizer(meshes, cameras=cameras)
        mask = (fragments.pix_to_face > 0).float()[0, :, :, 0:1]  # [n, y, x, k]

        maps = interpolate_face_attributes(
            fragments.pix_to_face,
            fragments.bary_coords,
            attributes,
        )

        mask = maps[0, :, :, 0, 0:3]

        return mask.permute(2, 0, 1).contiguous()

        # position_map = maps[0, :, :, 0, 0:3]
        # deforamtion_map = maps[0, :, :, 0, 3:]

        # return (
        #     position_map.permute(2, 0, 1).contiguous(),
        #     deforamtion_map.permute(2, 0, 1).contiguous(),
        #     mask.permute(2, 0, 1).contiguous(),
        #     fragments.pix_to_face[0].permute(2, 0, 1).contiguous(),
        # )
