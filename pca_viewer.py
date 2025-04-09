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
import sys
import numpy as np
import torch as th
import torch.utils.data
from pathlib import Path
from omegaconf import OmegaConf
import torchvision as tv
from tqdm import tqdm
import ffmpeg
from data.base import DatasetMode
from gaussians.renderer import render
from lib.apperance.pca_gaussian import PCApperance
from utils.geometry import AttrDict
from utils.general import build_dataset, build_loader, get_single, seed_everything, to_device, to_tensor
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix, quaternion_multiply
import torch.nn.functional as F
from loguru import logger
from scipy.spatial.transform import Rotation as R

from utils.text import write_text

torch.backends.cudnn.benchmark = True


def parse_payload(results, root_RT):
    means3D = results.geometry
    opacity = results.opacity
    scales = results.scales
    rotation = F.normalize(results.rotation)

    R = root_RT[:3, :3]
    T = root_RT[:3, 3]
    means3D = (R @ means3D.T).T + T

    N = means3D.shape[0]
    Q = matrix_to_quaternion(R)[None].expand(N, -1)
    rotation = quaternion_multiply(Q, rotation)

    pkg = {
        "means3D": means3D,
        "scales": scales,
        "rotation": rotation,
        "opacity": opacity,
    }

    if "shs" in results:
        pkg["shs"] = th.cat([results.colors[:, None, :], results.shs.reshape(-1, 15, 3)], dim=1)
        pkg["sh_degree"] = 3
    elif "shadow" in results:
        pkg["colors_precomp"] = results.apperance * (1.0 - th.clamp(results.shadow, 0, 1))
    else:
        pkg["colors_precomp"] = results.apperance

    return pkg


def splat(single, coeffs, pca, RT=None):
    results = pca.inverse_transform(coeffs)
    render_pkg = parse_payload(results, RT)

    pkg = render(single, render_pkg, bg_color="white", training=False)

    pred_image = pkg["render"]

    return pred_image


def get_frames(loader, target=100):
    frames = []
    logger.info(f"Loading {target} frames...")
    for j, batch in tqdm(enumerate(loader)):
        batch = to_device(batch)
        single = get_single(batch, 0)
        frames.append(single)
        if j >= target - 1:
            break

    return frames


def save_image(pred_image, path, info):
    mode = info["mode"]
    v = info["value"]
    value = f"{v:.2}".zfill(3)
    component = str(info["component"]).zfill(2)
    it = info["it"]
    mask = info["mask"]
    msg = f"{mode.upper()}/{mask} COMP {component} = {value}"

    print(msg)

    pred_image = write_text(pred_image, msg, fontColor=(0, 0, 0), set_W=300, bottom=True)
    frame_id = str(it).zfill(4)
    path = path + f"/{frame_id}.png"
    tv.utils.save_image(pred_image, path)


def save_video(src, use_parts):
    name = Path(src).stem
    suffix = "_parts" if use_parts else ""
    dst = src + f"/../{name}{suffix}.mp4"
    src = src + "/*.png"
    outputs = ffmpeg.input(src, pattern_type="glob", r=10)
    ffmpeg.filter(outputs, "pad", width="ceil(iw/2)*2", height="ceil(ih/2)*2").output(
        dst,
        pix_fmt="yuv420p",
        crf=20,
    ).overwrite_output().run()


def show(config, use_parts):
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

    frames = get_frames(loader, 1)

    dst = config.train.results_dir + "/pca"
    # os.system(f"rm -rf {dst}")
    Path(dst).mkdir(parents=True, exist_ok=True)

    config.train.use_parts = use_parts

    pca = PCApperance(config).cuda()

    coeffs = {}
    for mask in pca.masks:
        coeffs[mask] = th.zeros([pca.n]).cuda()

    # for k, v in coeffs.items():
    #     print(f"{k}: {v.shape}")
    # exit(0)

    RT = frames[0]["root_RT"]

    if config.dataset_name == "NERSEMBLE":
        r = R.from_euler('y', 0, degrees=True).as_matrix()
        RT[:3, :3] = th.from_numpy(r).cuda().float()
        RT[:3, 3][0] = 0
        RT[:3, 3][1] = 0

        frames[0]["R"] = th.from_numpy(r).cuda().float()
        frames[0]["R"][:3, 1] *= -1
        frames[0]["R"][:3, 2] *= -1
        frames[0]["T"][0] = 0 
        frames[0]["T"][1] = 0 

    i = 0
    for m, mod in enumerate(pca.keys()):
        for mask in pca.masks:
            # Current mode
            offset = m * pca.n_components[mod]
            # Components
            for c in range(5):
                # Rendering values between -3std and 3std
                for s in np.linspace(-3, 3, num=10):
                    v = th.zeros([pca.n]).cuda()
                    v[offset + c] = s
                    coeffs[mask] = v

                    with th.no_grad():
                        image = splat(frames[0], pca.to_coeffs(coeffs), pca, RT)

                    info = {"mode": mod, "value": s, "component": c, "it": i, "mask": mask}

                    save_image(image, dst, info)
                    i += 1

    save_video(dst, use_parts)


if __name__ == "__main__":
    path = sys.argv[1]
    config = OmegaConf.load(path)
    seed_everything()

    show(config, True)
    show(config, False)
 