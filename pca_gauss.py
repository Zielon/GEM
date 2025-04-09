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


from pathlib import Path
import sys
import numpy as np
import torch as th
import torch.utils.data
from sklearn.decomposition import PCA
from omegaconf import OmegaConf
from loguru import logger
from tqdm import tqdm
from data.base import DatasetMode
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_axis_angle
from lib.apperance.trainer import ApperanceTrainer
from train import folders
from lib.F3DMM.masks.masking import Masking
from utils.general import build_dataset, build_loader, get_single, seed_everything, to_device

torch.backends.cudnn.benchmark = True


def to_fp16(obj):
    if isinstance(obj, dict):
        return {key: to_fp16(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [to_fp16(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.astype(np.float16)
    else:
        return obj


def remove_joint(xyz, A, W, tex_to_mesh, joint=0):
    xyz = xyz[None]

    A[:, joint, ...] = th.linalg.inv(A[:, joint, ...])

    for i in range(5):
        if i == joint:
            W[:, :, i] = 1
        else:
            W[:, :, i] = 0

    T = th.matmul(W, A.view(1, 5, 16)).view(1, -1, 4, 4)
    T = T[:, tex_to_mesh, ...]

    homogen_coord = th.ones([1, xyz.shape[1], 1]).cuda()
    v_posed_homo = th.cat([xyz, homogen_coord], dim=2)
    v_homo = th.matmul(T, th.unsqueeze(v_posed_homo, dim=-1))

    xyz = v_homo[:, :, :3, 0]

    return xyz[0]


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    rot_mats = quaternion_to_matrix(quaternions)
    axis_angles = matrix_to_axis_angle(rot_mats)
    return axis_angles


def merge_textures(pkg, single, tex_to_mesh):
    if "pred" in pkg:
        pkg = pkg.pred

    geometry = pkg.means3D

    geometry = remove_joint(geometry, single["A"], single["W"], tex_to_mesh, joint=1)

    payload = {
        "geometry": geometry,
        "opacity": pkg.opacity,
        "scales": pkg.scales,
        # "rotation": quaternion_to_axis_angle(pkg.rotation),
        "rotation": pkg.rotation,
        "colors": pkg.colors_precomp
    }

    return payload


def save_xyz(filename, pc):
    pc = pc.cpu().numpy()
    np.savetxt(filename, pc, fmt='%.6f', delimiter=' ', header='x y z', comments='')


def run(config, quantize=False, use_parts=False):
    config.data.join_configs = True
    dataset = build_dataset(config, camera_list=[config.data.test_camera], mode=DatasetMode.validation)
    loader = build_loader(
        dataset,
        batch_size=1,
        num_workers=10,
        shuffle=False,
        persistent_workers=True,
        seed=33,
    )

    trainer = ApperanceTrainer(config, dataset)
    trainer.restore()
    trainer.eval()
    masking = Masking()

    # Disable Gaussian masking
    trainer.model.gaussian_mask = None

    list_masks = {"full": masking.full().cuda()}
    if use_parts:    
        list_masks = masking.list_masks()

    uv_size = config.train.uv_size
    tex_to_mesh = trainer.model.tex_to_mesh

    Path(f"checkpoints/GAUSSIAN_PCA_{uv_size}").mkdir(exist_ok=True, parents=True)
    logger.info(f"Building PCA with total of {len(dataset)} frames")

    rgb = None
    gaussians = {}

    # 1) Build Gaussian dataset

    j = 0
    with th.no_grad():
        for batch in tqdm(loader):
            single = get_single(to_device(batch), 0)
            pkg = trainer.model.predict(single, to_canonical=True)
            merged = merge_textures(pkg, single, tex_to_mesh)
            for k in merged.keys():
                if k not in gaussians:
                    gaussians[k] = []
                gaussians[k].append(merged[k])

            if rgb is None:
                rgb = merged["colors"]

            j += 1

    # 2) Build PCA

    config_pca = {
        "geometry": 50,
        "opacity": 50,
        "scales": 40,
        "rotation": 40,
    }

    coeffs = {}
    bases = {"colors": {}, "tex_map": {}}

    for name, map in gaussians.items():
        if name == "colors":
            continue
        per_part_bases = {}
        per_part_coeffs = {}
        per_part_map = {}
        for part_name, part_mask in list_masks.items():
            mask = part_mask[tex_to_mesh]

            Mat = th.stack(map)
            N = len(Mat)
            C = Mat.shape[2]

            Mat = Mat[:, mask, :]
            # if name == "geometry":
            #     save_xyz(f"pc_{part_name}.xyz", Mat[0])
            Mat = Mat.reshape(N, -1).cpu().numpy()

            pca = PCA(n_components=config_pca[name])
            pca.fit(Mat)

            modality = {"components": pca.components_, "variance": pca.explained_variance_, "mean": pca.mean_, "channels": C}

            if "part_name" not in bases["colors"]:
                bases["colors"][part_name] = rgb[mask]
                bases["tex_map"][part_name] = tex_to_mesh[mask]

            per_part_bases[part_name] = modality
            per_part_coeffs[part_name] = pca.transform(Mat)

            logger.info(f"PCA for {name}/{part_name} with {config_pca[name]} was created...")

        bases[name] = per_part_bases
        coeffs[name] = per_part_coeffs

    ###############################

    bases["config"] = config_pca

    suffix = "_parts" if use_parts else ""

    if quantize:
        bases = to_fp16(bases)

    th.save(bases, f"checkpoints/GAUSSIAN_PCA_{uv_size}/{config.capture_id}{suffix}.ptk")
    th.save(coeffs, f"checkpoints/GAUSSIAN_PCA_{uv_size}/{config.capture_id}_coeffs{suffix}.ptk")


if __name__ == "__main__":
    path = sys.argv[1]
    config = OmegaConf.load(path)

    seed_everything()
    folders(config)

    quantize = False

    run(config, quantize, use_parts=False)
    run(config, quantize, use_parts=True)
