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


import json
import math
from pathlib import Path
import sys
import cv2
import numpy as np
import torch as th
import torch.utils.data
import itertools

import trimesh
from omegaconf import OmegaConf
from loguru import logger
from tqdm import tqdm
from data.base import DatasetMode
from lib.apperance.trainer import ApperanceTrainer
from lib.regressor.trainer import RegressorTrainer
from train import folders
import torchvision as tv
import torch.nn.functional as F
from utils.error import compute_errors, compute_heatmap
from utils.general import build_dataset, build_loader, get_single, seed_everything, to_device
from utils.renderer import Renderer
from utils.text import write_text
from concurrent.futures import ThreadPoolExecutor
import ffmpeg
from scipy.spatial.transform import Rotation as Rscipy
from utils.timers import cuda_timer

torch.backends.cudnn.benchmark = True

DST_NAME = "test"

rasterize = Renderer(white_background=False).cuda()
rasterize.resize(1024, 667)


def interpolate(t, s):
    return F.interpolate(t[None], (s, s), mode="bilinear")[0]


def infinite_iterator(lst):
    ordering = lst
    while True:
        for item in ordering:
            yield item
        ordering = ordering[::-1]


def rotate_R(R: torch.Tensor, axis: str, angle_deg: float) -> torch.Tensor:
    axis_map = {
        'x': [1, 0, 0],
        'y': [0, 1, 0],
        'z': [0, 0, 1]
    }

    if axis.lower() not in axis_map:
        raise ValueError("Axis must be one of 'x', 'y', or 'z'")

    axis_vec = np.array(axis_map[axis.lower()])
    r = Rscipy.from_rotvec(np.deg2rad(angle_deg) * axis_vec)
    rot_np = r.as_matrix()

    rot = torch.tensor(rot_np, device=R.device, dtype=R.dtype)
    return rot @ R


def central_paste(image, bg_color, dH, dW):
    C, H, W = image.shape
    device = image.device
    if H != dH or W != dW:
        bg = th.zeros([3, dH, dW]).to(device) if bg_color == "black" else th.ones([3, dH, dW]).to(device)
        h = (dH - H) // 2
        w = (dW - W) // 2
        bg[:, h : h + H, w : w + W] = image
        return bg
    return image



def generate_cameras_world_space(R, T, lookat, n_cameras, angle_range=(-40, 40)):
    R, T, lookat = R.float(), T.float(), lookat.float()
    camera_position = -(R.T @ T)  # Compute camera position in world space
    angles = np.radians(np.linspace(angle_range[0], angle_range[1], n_cameras))
    cameras = []

    for angle in angles:
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_y = torch.tensor([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], device=R.device, dtype=R.dtype)

        # Rotate camera position around the lookat point
        translated_position = camera_position - lookat
        rotated_position = rotation_y @ translated_position
        new_camera_position = rotated_position + lookat

        # Compute forward vector (lookat to new_camera_position)
        forward = torch.nn.functional.normalize(new_camera_position - lookat, dim=0)

        # Compute right and up vectors
        up_vector = torch.tensor([0, 1, 0], device=R.device, dtype=R.dtype)
        right = torch.nn.functional.normalize(torch.linalg.cross(up_vector, forward), dim=0)
        up = torch.linalg.cross(forward, right)

        # Compute world-space rotation matrix
        world_R = torch.stack([right, -up, -forward], dim=1)

        # Translation remains as new_camera_position
        world_T = -world_R @ new_camera_position

        # Store the world-space camera
        cameras.append((world_R, world_T))

    return cameras


def save_image(pkg, config, errors, iter, is_spin):
    gt_image = pkg.gt_image
    alpha = pkg.alpha
    pred_alpha = pkg.pred_alpha
    pred_image = pkg.pred_image
    cam_id = pkg.cam_id
    mesh_rendering = pkg.mesh_rendering
    vis = pkg.vis

    msg = "RGB (CNN)" if not config.train.get("use_sh", False) else "DC (SH)"
    C, H, W = pred_image.shape

    gt_image = central_paste(gt_image, config.train.bg_color, H, W)

    if pkg.additional is not None:
        gt_image = central_paste(pkg.additional, config.train.bg_color, H, W)

    split = DST_NAME.split("_TO_") if "_TO_"in DST_NAME else None

    heapmap = None
    if "_TO_" not in DST_NAME or (split and split[0] == split[1]):
        heapmap = th.from_numpy(compute_heatmap(gt_image, pred_image)).permute(2, 0, 1).cuda()

    H = H // 2
    # canon_normals = interpolate(vis.canonical_normals, H)
    # def_normals = interpolate(vis.deformed_normals, H)
    # colors = interpolate(vis.colors, H)
    # if vis.image is not None:
    #     image = interpolate(vis.image, H)
    # else:
    #     image = th.zeros_like(colors)

    fc = (1, 1, 1) if config.train.bg_color == "black" else (0, 0, 0)
    # col_0 = th.cat([(canon_normals + 1) * 0.5, colors], dim=1)
    # col_1 = th.cat([(def_normals + 1) * 0.5, image], dim=1)

    n_pca = config.train.get("pca_n_components", 0)
    exp_name = config.train.get("exp_name", "noname").replace("_", " ").upper()

    mesh_msg = f"Input Mesh (PCA {n_pca})"

    if "transfer" not in DST_NAME:
        mesh_msg = "EMOCA" if config.train.get("trainer", "") == "regressor" else "MESH"

    images = [
        write_text(write_text(gt_image, "Ground Truth", fc), exp_name, fc, bottom=True),
        write_text(mesh_rendering, mesh_msg, fc),
        write_text(pred_image, f"Pred (N={vis.n_gaussian})", fc),
    ]

    if heapmap is not None:
        images += [heapmap]

    # images += [
    #     write_text(write_text(col_0, msg, set_W=667, bottom=True), f"Canonical", set_W=667),
    #     write_text(col_1, "Deformed", set_W=667),
    # ]

    grid = th.cat(images, dim=2)

    frame_id = str(iter).zfill(4)
    path = config.train.results_dir + f"/{DST_NAME}/{frame_id}.png"

    C, H, W = grid.shape
    # grid = F.interpolate(grid[None], (H // 2, W // 2), mode="bilinear")[0]
    # tv.utils.save_image(grid, path.replace("png", "jpg"))

    if heapmap is not None:
        metrics = {}
        metrics = compute_errors(gt_image, pred_image)
        errors[frame_id] = metrics

    # pred_image = pred_alpha * pred_image + (pred_alpha - 1.0) * th.ones_like(pred_image)

    folder_name = "spin" if is_spin else "renders"

    if "_TO_" in DST_NAME:
        pred_image = th.cat([gt_image, pred_image], dim=2)

    path = config.train.results_dir + f"/{folder_name}_{DST_NAME}/{frame_id}_{cam_id}.png"
    tv.utils.save_image(pred_image, path)

    if not is_spin:
        path = config.train.results_dir + f"/gt_{DST_NAME}/{frame_id}_{cam_id}.png"
        tv.utils.save_image(gt_image, path)


def save_mesh(pkg):
    frame = pkg["single"]["frame"]
    vertices = pkg["single"]["geom_vertices"].float()
    faces = pkg["single"]["geom_faces"].long()

    dst = f"tmp/{frame}.ply"

    v = vertices.cpu().numpy()
    f = faces.cpu().numpy()
    m = trimesh.Trimesh(v, f, process=False)

    m.export(dst)


def save_video(config, is_spin, fps=25):
    target_id = config.capture_id
    folder_name = "spin" if is_spin else "renders"
    src = config.train.results_dir + f"/{folder_name}_{DST_NAME}/*.png"
    dst = config.train.results_dir + f"/{folder_name}_{DST_NAME}.mp4"
    outputs = ffmpeg.input(src, pattern_type="glob", r=fps)
    ffmpeg.filter(outputs, "pad", width="ceil(iw/2)*2", height="ceil(ih/2)*2").output(
        dst,
        pix_fmt="yuv420p",
        crf=25,
    ).overwrite_output().run()


def save_errors(errors, dst, name="errors"):
    psnr = 0
    lipis = 0
    l1 = 0
    ssim = 0
    i = 0

    for key, item in errors.items():
        psnr += item["psnr"]
        lipis += item["lpips"]
        l1 += item["l1"]
        ssim += item["ssim"]
        i += 1

    i = max(i, 1)

    psnr /= i
    lipis /= i
    ssim /= i
    l1 /= i

    dst = dst + f"/{name}.json"

    errors = {
        "PSNR": psnr,
        "LIPIS": lipis,
        "L1": l1,
        "SSIM": ssim,
    }

    with open(dst, "w") as file:
        json.dump(errors, file, indent=4, sort_keys=True)


def lunch(dataset, config, spin_list=[False, True], angle=40):
    logger.info(f"TEST with mode {DST_NAME}")
    fps = config.get("fps", 25)
    loader = build_loader(
        dataset,
        batch_size=1,
        num_workers=8,
        shuffle=False,
        persistent_workers=True,
        seed=33,
    )

    rasterize.set_bg_color([1, 1, 1] if config.train.get("bg_color", "black") == "white" else [0, 0, 0])

    is_regressor = False
    if config.get("regressor", None) is None:
        trainer = ApperanceTrainer(config, dataset)
    else:
        is_regressor = True
        config.train.use_data_augmentation = False
        trainer = RegressorTrainer(config, dataset)

    trainer.restore()
    trainer.eval()
    trainer.model.bg_color = "white"

    logger.info(f"Test with total of {len(dataset)} frames")

    Path(config.train.results_dir + f"/spin_{DST_NAME}").mkdir(parents=True, exist_ok=True)
    if False in spin_list:
        Path(config.train.results_dir + f"/renders_{DST_NAME}").mkdir(parents=True, exist_ok=True)
        Path(config.train.results_dir + f"/gt_{DST_NAME}").mkdir(parents=True, exist_ok=True)

    errors = {}

    R = torch.tensor([[ 1.,  0.,  0.], [ 0.,  -1.,  0.], [ 0.,  0.,  -1.]]).cuda().float()
    T = torch.tensor([0.0, 0.0, 1.1]).cuda().float()
    center = torch.tensor([0,  0, -0.07], device='cuda:0', dtype=torch.float32)
    cameras = infinite_iterator(generate_cameras_world_space(R, T, center, 100, angle_range=(-20, 20)))

    r_y = Rscipy.from_euler('y', 0, degrees=True).as_matrix()
    r_x = Rscipy.from_euler('x', 7, degrees=True).as_matrix()
    r_z = Rscipy.from_euler('z', 0, degrees=True).as_matrix()

    with th.no_grad():
        for is_spin in spin_list:
            for j, batch in tqdm(enumerate(loader)):
                batch = to_device(batch)
                single = get_single(batch, 0)
        
                gt_image = single["image"]
                alpha = single["alpha"]

                # Rotating camera
                if is_spin:
                    nR, nT = next(cameras)
                    single["R"] = nR
                    single["T"] = nT
                    batch["R"] = nR[None]
                    batch["T"] = nT[None]
                    if "_TO_" in DST_NAME:
                        batch["root_RT"][:, :3, :3] = th.from_numpy(r_x @ r_y @ r_z).cuda().float()

                C, H, W = gt_image.shape
                rasterize.resize(H, W)
                trainer.renderer.resize(H, W)

                ############################################################

                input = batch if is_regressor else single
                vis, _ = trainer.inference(input)

                ############################################################

                bg = th.zeros_like(gt_image) if trainer.bg_color == "black" else th.ones_like(gt_image)
                vis.gt_image = gt_image * alpha + bg * (1 - alpha)
                vis.alpha = alpha
                vis.additional = None
                if "additional" in single:
                    vis.additional = single["additional"]
                save_image(vis, config, errors, j, is_spin)

                # if j > 1000:
                #     break

            save_video(config, is_spin, fps=fps)
            if not is_spin:
                save_errors(errors, config.train.results_dir, name=f"errors_{DST_NAME}")


def validation(config):
    global DST_NAME
    DST_NAME = "val"
    config.data.join_configs = False
    dataset = build_dataset(config, camera_list=[config.data.test_camera], mode=DatasetMode.validation)
    lunch(dataset, config)


def test(config):
    global DST_NAME
    DST_NAME = "test"
    config.data.join_configs = False
    dataset = build_dataset(config, camera_list=[config.data.test_camera], mode=DatasetMode.test)
    lunch(dataset, config)


def reeanacment(config, source_config, mode):
    target = config.capture_id
    is_deca = False
    if "mp4" not in source_config:
        soruce = source_config.capture_id
        is_deca = source_config.get("source_id", None) != None
        spin_list=[False]
    else:
        soruce = source_config.split("/")[-1].replace(".mp4", "")
        spin_list=[True]
    global DST_NAME
    DST_NAME = (f"{soruce}_to_{target}" + ("_DECA" if is_deca else "")).upper()
    config.data.join_configs = True
    dataset = build_dataset(config, camera_list=[config.data.test_camera], source_config=source_config, mode=mode)

    lunch(dataset, config, spin_list, angle=20)


if __name__ == "__main__":
    traget_config = OmegaConf.load(sys.argv[1])
    traget_config.train.batch_size = 1
    traget_config.train.bg_color = "white"

    seed_everything()
    folders(traget_config)

    if len(sys.argv) >= 3:
        if ".mp4" in sys.argv[2]:
            traget_config.fps = 20  # change fps for video
            reeanacment(traget_config, sys.argv[2], DatasetMode.video)
            exit(0)
        else:
            source_config = OmegaConf.load(sys.argv[2])
            reeanacment(traget_config, source_config, DatasetMode.validation)

    test(traget_config)
    validation(traget_config)
