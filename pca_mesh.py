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
from sklearn.decomposition import PCA
from utils.general import build_dataset, build_loader, get_single, seed_everything, to_device

torch.backends.cudnn.benchmark = True
rasterize = Renderer(white_background=False).cuda()


def rasterize_map(camera, mesh, def_grad, canonical_mesh):
    with th.no_grad():
        deformed = mesh[0][None].cuda()
        canonical = canonical_mesh[0][None].cuda()
        faces = mesh[1][None].long().cuda()

        maps = rasterize.map(camera, deformed, canonical, def_grad, faces)

        position_map = maps[0]
        deformation_map = maps[1]
        mask = maps[2]
        pix_to_face = maps[3]

        return position_map[None], deformation_map[None], mask[None], pix_to_face[None]


def get_pos_map(single, canonical_mesh):
    mesh = single["geom_vertices"].float(), single["geom_faces"].long()
    rasterize.resize(512, 512)
    to_local, to_deform = deformation_gradient(mesh, canonical_mesh)
    J = to_deform @ to_local
    pos_map, _, mask, pix_to_face = rasterize_map(rasterize.front_camera, mesh, J, canonical_mesh)
    return pos_map.cpu(), mask.cpu(), pix_to_face.cpu()


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


def save_detection_results(config, detection_results):
    capture_id = config.capture_id
    prefix = "/home/wzielonka/Cluster/lustre/fast" if not os.path.exists("/fast/wzielonka/") else ""
    dst = f"{prefix}/fast/wzielonka/datasets/volumetric-tracker/{capture_id}/"
    Path(dst).mkdir(parents=True, exist_ok=True)
    Path(dst, "lmk").mkdir(parents=True, exist_ok=True)

    points = []
    for results, frame in detection_results:
        canonicals = []
        if len(results.face_landmarks) > 0:
            for lmk in results.face_landmarks[0]:
                canon = np.array([lmk.x, lmk.y, lmk.z, 1])
                # Rt = np.array(results.facial_transformation_matrixes[0])
                # posed = Rt @ canon
                canonicals.append(canon)
            lmks = np.array(canonicals)[:, 0:3]
            points.append(lmks)
            np.save(f"{dst}/lmk/{frame}.npy", lmks)
        else:
            logger.info(f"Frame {frame} has no landmarks!")

    points = np.array(points)
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    th.save({"mean": mean, "std": std}, f"{dst}/lmk_stats.pt")


def generate(config):
    config.data.join_configs = True
    dataset = build_dataset(config, camera_list=[config.data.test_camera], mode=DatasetMode.train)
    loader = build_loader(dataset, shuffle=False, num_workers=8, batch_size=1)

    logger.info(f"Creating PCA using {len(loader)} meshes...")

    if len(loader) == 0:
        raise ValueError("No meshes found!")

    maps = []
    for j, batch in tqdm(enumerate(loader)):
        batch = to_device(batch)
        single = get_single(batch, 0)
        C, H, W = single["image"].shape
        rasterize.resize(H, W)
        maps.append(single["geom_vertices"].float().cpu().numpy())

    M = np.array(maps)
    B, V, C = M.shape

    M_flat = M.reshape(B, -1)
    pca = PCA(n_components=100, whiten=True)
    pca.fit(M_flat)

    checkpoint = {
        "components": pca.components_,
        "variance": pca.explained_variance_,
        "mean": pca.mean_,
    }

    os.makedirs("checkpoints/MESH_PCA", exist_ok=True)
    th.save(checkpoint, f"checkpoints/MESH_PCA/{config.capture_id}_mesh.ptk")

    # Test
    vertices = th.from_numpy(maps[0])

    pca_layer = PCALayer(pca.components_, pca.mean_, pca.explained_variance_).cpu()
    nearest_pca_layer = pca_layer(vertices)

    # Test
    a = M_flat[0][None]
    coeffs = pca.transform(a)

    coeffs_layer = pca_layer.transform(vertices.reshape(-1))
    nearest = pca.inverse_transform(coeffs).reshape(1, V, C)

    error = np.mean((nearest.reshape(V, C) - nearest_pca_layer.numpy().reshape(V, C)) ** 2)

    logger.info(f"Error = {error:.8f}")

if __name__ == "__main__":
    seed_everything()
    for path in sorted(glob(f"configs/nersemble/*/default.yml")):
        logger.info(f"Processing {path}")
        config = OmegaConf.load(path)
        generate(config)
