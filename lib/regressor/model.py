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
from glob import glob
from pathlib import Path
import cv2
import numpy as np
import torch as th
import trimesh
from utils.general import get_single
import torch.nn.functional as F
from collections import deque
from encoder.encoder import ResnetEncoder
from gaussians.renderer import render
from lib.F3DMM.FLAME2020.flame import FLAME
from lib.apperance.model import ApperanceModel
from lib.apperance.pca_gaussian import PCApperance
from lib.base_model import BaseModel
from lib.common import Mesh
from data.transfer import TransferDataset
from loguru import logger
from omegaconf import OmegaConf
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_quaternion, quaternion_to_matrix, quaternion_multiply
from utils.geometry import (
    AttrDict,
)


class RegressorModel(BaseModel):
    def __init__(self, config, dataset) -> None:
        super().__init__(config, dataset)
        self.apperance = PCApperance(config).cuda()
        self.resnet = ResnetEncoder(self.apperance.n, config, dataset).cuda()
        self.flame = FLAME().cuda().eval()
        self.gaussian_model = ApperanceModel(self.config, dataset)
        self.is_eval = False
        self.refine_basis = False
        self.inactive_gem_mask = None
        self.is_reenactment = type(dataset) is TransferDataset
        if self.is_reenactment:
            logger.warning("Reenactment mode is turned on!")

    def eval(self):
        self.is_eval = True
        self.resnet.eval()
        self.resnet.to_jit()
        if type(self.dataset) is TransferDataset:
            self.resnet.source_features = self.resnet.get_canonical_features(self.dataset.source.identity_frame)

    def reset_running_bbox(self):
        self.resnet.current_bbox = None
        self.resnet.running_window = deque(maxlen=self.resnet.windows_size)

    def apply_neck(self, single, xyz, tex_to_mesh):
        xyz = xyz[None]
        A = single["A"]
        W = single["W"]

        W[:, :, 0] = 0
        W[:, :, 1] = 1
        W[:, :, 2] = 0
        W[:, :, 3] = 0
        W[:, :, 4] = 0

        T = th.matmul(W, A.view(1, 5, 16)).view(1, -1, 4, 4)
        T = T[:, tex_to_mesh, ...]

        homogen_coord = th.ones([1, xyz.shape[1], 1]).cuda()
        v_posed_homo = th.cat([xyz, homogen_coord], dim=2)
        v_homo = th.matmul(T, th.unsqueeze(v_posed_homo, dim=-1))

        xyz = v_homo[:, :, :3, 0]

        return xyz[0]

    def parameters(self):
        return [self.resnet.parameters(), self.apperance.parameters()]

    def count_parameters(self):
        for model_name in ["apperance", "resnet"]:
            if hasattr(self, model_name):
                model = getattr(self, model_name)
                n = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"{str(type(model).__name__).ljust(20, ' ')} parameters={n}")

    def create(self):
        pass

    def load_state_dict(self, state):
        resnet, apperance = state
        self.resnet.load_state_dict(resnet)
        self.apperance.load_state_dict(apperance)

    def state_dict(self):
        return (self.resnet.state_dict(), self.apperance.state_dict())

    def get_opt_params(self):
        return self.resnet.parameters()

    def step(self, curr_iter):
        pass

    def flame_mesh(self, codedict):
        pose = codedict["pose"]
        pose[:, :3] = 0
        verts, _, _ = self.flame(shape_params=codedict["shape"] * 0, expression_params=codedict["exp"], pose_params=codedict["pose"])
        return verts[0]

    def parse_payload(self, results, root_RT):
        means3D = results.geometry
        opacity = results.opacity
        scales = results.scales
        rotation = F.normalize(results.rotation)

        if root_RT is not None:
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

    def splat(self, single, results, to_canonical=False):
        root_RT = single["root_RT"]

        if to_canonical:
            root_RT = None

        render_pkg = self.parse_payload(results, root_RT)

        pkg = render(single, render_pkg, bg_color=self.bg_color)

        pred_image = pkg["render"]
        pred_alpha = pkg["alpha"]
        bg_color = pkg["bg_color"]

        return pred_image, render_pkg, pred_alpha, bg_color

    def to_gaussian_maps(self, results):
        with th.no_grad():
            colors = results.apperance.reshape(1, self.uv_size, self.uv_size, 3).permute(0, 3, 1, 2)
            means3D = results.geometry.reshape(1, self.uv_size, self.uv_size, 3).permute(0, 3, 1, 2)
            opacity = results.opacity.reshape(1, self.uv_size, self.uv_size, 1).permute(0, 3, 1, 2)
            scales = results.scales.reshape(1, self.uv_size, self.uv_size, 3).permute(0, 3, 1, 2)
            rotation = results.rotation.reshape(1, self.uv_size, self.uv_size, 4).permute(0, 3, 1, 2)

            maps = {
                "scales": scales,
                "rotation": rotation,
                "position": means3D,
                "rgb": colors,
                "opacity": opacity,
            }

        return AttrDict(maps)

    def parse(self, pca):
        preds = {}
        n = 0
        t = self.apperance.n 
        for key in self.apperance.masks:
            preds[key] = pca[:, n : n + t]
            n += t

        return self.apperance.to_coeffs(preds, is_batch=True)

    def slice(self, pred_codes, b):
        preds = {}
        for k in pred_codes.keys():
            preds[k] = pred_codes[k][b]
        return preds

    def stack(self, list_dict):
        merged = {}
        for dict in list_dict:
            if not dict:
                continue
            for k in dict.keys():
                if k not in merged:
                    merged[k] = []
                merged[k].append(dict[k])

        for k in merged.keys():
            merged[k] = th.stack(merged[k])

        return merged

    def enable_region(self, coeffs, name):
        for k in coeffs.keys():
            if name in k:
                continue
            coeffs[k] *= 0.0
        return coeffs

    def predict(self, batch, is_warmup=False):
        gt_codes = None

        # gt_R = matrix_to_axis_angle(single["root_RT"][:3, :3])
        # gt_T = single["root_RT"][:3, 3]

        N = self.pca_n_components

        #### PREDCITION ####

        preds = self.resnet(batch, reenactment=self.is_reenactment)

        if preds is None:
            return None, None

        raw_codes = preds.pca
        paresed_preds = self.parse(raw_codes)

        deca = {}
        deca["pred_codes"] = paresed_preds
        deca["raw_codes"] = raw_codes
        deca["gt_codes"] = []
        deca["bbox"] = preds.bbox

        render_pkg_list = []
        B = batch["image"].shape[0]
        for b in range(B):
            single = get_single(batch, b)
            pred_codes = self.slice(paresed_preds, b)

            if not self.is_eval:
                gt_codes = self.apperance.get(single["frame"].item())

            if is_warmup and not self.is_eval:
                pred_codes = gt_codes

            deca["gt_codes"].append(gt_codes)

            vertices = self.flame_mesh(preds.deca[b])
            # mesh = Mesh(single["geom_vertices"].float(), single["geom_faces"].long())
            mesh = Mesh(vertices, batch["geom_faces"][0].long())

            #### DECA DEBUG ####
            # frame_id = single["frame"].item()
            # camera_id = single["cam_idx"]
            # gt_image = single["image"]
            # alpha = single["alpha"]
            # gt_image = gt_image * alpha + (1 - alpha)
            # Path("debug").mkdir(parents=True, exist_ok=True)
            # cv2.imwrite(f"debug/{frame_id}_{camera_id}_cropped.png", preds.deca_input[0][0].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu().numpy() * 255)
            # cv2.imwrite(f"debug/{frame_id}_{camera_id}_input.png", gt_image.permute(1, 2, 0)[:, :, [2, 1, 0]].cpu().numpy() * 255)
            # trimesh.Trimesh(vertices.detach().cpu().numpy(), self.flame.faces_tensor.detach().cpu().numpy()).export(f"debug/{frame_id}_{camera_id}.ply")

            masking = None
            if self.is_eval:
                masking = {
                    # For multipart regression
                    "keys": ["scalp", "neck"],
                    # For inactive regions
                    "mask": self.inactive_gem_mask
                }

            # pred_codes = self.enable_region(pred_codes, "mouth")

            results = self.apperance.inverse_transform(pred_codes, masking=masking)

            if ("A" in single and "W" in single) and not self.is_eval:
                results.geometry = self.apply_neck(single, results.geometry, results.tex_map)

            pred_image, render_pkg, pred_alpha, bg_color = self.splat(single, results)

            # For visualziation
            render_pkg["pred_image"] = pred_image
            render_pkg["pred_alpha"] = pred_alpha
            render_pkg["image"] = preds.deca_input[b][0]
            render_pkg["mesh"] = mesh
            render_pkg["canonical_state"] = self.gaussian_model.canonical_state
            render_pkg["gaussian"] = results
            render_pkg["n_gaussian"] = self.uv_size**2
            render_pkg["bg_color"] = bg_color

            render_pkg_list.append(AttrDict(render_pkg))
        
        deca["gt_codes"] = self.stack(deca["gt_codes"])

        return AttrDict(deca), render_pkg_list
