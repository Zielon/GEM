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
from pathlib import Path
import torchvision as tv
from re import T
import sys
from tkinter.tix import Tree
import cv2
import torch as th
import torch.utils.data
from glob import glob
from data.base import DatasetMode
from utils.pca_layer import PCALayer
from utils.renderer import Renderer
from utils.geometry import calculate_tbn, deformation_gradient
from omegaconf import OmegaConf
from loguru import logger
from tqdm import tqdm
import numpy as np
import ffmpeg
import trimesh
from sklearn.decomposition import PCA
from utils.general import build_dataset, build_loader, get_single, seed_everything, to_device

torch.backends.cudnn.benchmark = True
rasterize = Renderer(white_background=False).cuda()
mesh = trimesh.load("assets/meshes/flame_mask.obj", process=False)
face_colors = mesh.visual.vertex_colors[:, :3]

red_color = np.array([255, 0, 0])
color_mask = np.all(face_colors == red_color, axis=1).astype(int)
color_mask = color_mask[mesh.faces]
color_mask = th.tensor(color_mask).cuda().float()[..., None]


def rasterize_map(camera, mesh):
    with th.no_grad():
        deformed = mesh[0][None].cuda().float()
        faces = mesh[1][None].long().cuda()

        mask = rasterize.map(camera, deformed, faces, color_mask)

        return mask[None]


def render_mesh(pkg, mesh):
    with th.no_grad():
        root_RT = pkg["root_RT"]
        R = root_RT[:3, :3]
        T = root_RT[:3, 3]

        cameras = Renderer.to_cameras(pkg)
        vertices = mesh[0].float()
        vertices = (R @ vertices.T).T + T
        faces = mesh[1].long()[None]
        mesh_rendering = rasterize(cameras, vertices[None], faces).permute(2, 0, 1)

        return mesh_rendering


def extract(pos_map, mask):
    B, C, H, W = mask.shape
    w, h = th.meshgrid(
        th.arange(H, device=pos_map.device),
        th.arange(W, device=pos_map.device),
        indexing="xy",
    )

    idsH = h[mask[0][0] > 0]
    idsW = w[mask[0][0] > 0]

    extracted = pos_map[:, :, idsH, idsW][0].permute(1, 0)

    return extracted

def generate(config, detector):
    camera_list=[config.data.test_camera]
    dataset = build_dataset(config, camera_list=None, mode=DatasetMode.train)
    loader = build_loader(dataset, shuffle=False, num_workers=8, batch_size=1)
    canonical_mesh = dataset.get_canonical_mesh()

    for j, batch in tqdm(enumerate(loader)):
        batch = to_device(batch)
        single = get_single(batch, 0)
        C, H, W = single["image"].shape
        rasterize.resize(H, W)

        mesh = (single["geom_vertices"], single["geom_faces"])

        alpha = single["alpha"]
        image = single["image"]
        image_path = single["image_path"]

        rendering = (1 - rasterize_map(Renderer.to_cameras(single), mesh)) * alpha
        # rendering = rasterize_map(Renderer.to_cameras(single), mesh)

        Path(image_path.replace("images", "alpha")).parent.mkdir(parents=True, exist_ok=True)

        tv.utils.save_image(rendering, image_path.replace("images", "alpha").replace(".jpg", ".png"))

if __name__ == "__main__":
    seed_everything()
    for path in sorted(glob(f"configs/nersemble/GCZ208/default.yml")):
        logger.info(f"Processing {path}")
        config = OmegaConf.load(path)
        generate(config, None)
