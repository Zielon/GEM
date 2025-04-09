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


import copy
from glob import glob
import os
import numpy as np
import torch as th
import trimesh
from lib.F3DMM.masks.masking import Masking
from lib.apperance.model import ApperanceModel
from lib.base_trainer import BaseTrainer
import torchvision as tv
from tqdm import tqdm
from loguru import logger
from gaussians.losses import VGGLoss, VGGPerceptualLoss, l1_image_grad_loss
from kornia.filters import gaussian_blur2d
from gaussians.losses import l1_loss, ssim
from gaussians.utils import SH2RGB
from lib.common import Mesh, interpolate, to_trimesh
from utils.error import compute_heatmap
from utils.general import get_single, instantiate
from pytorch3d.structures import Meshes
from utils.renderer import Renderer
from utils.text import write_text
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils.geometry import (
    AttrDict,
    GeometryModule,
    calculate_tbn_uv,
)


class ApperanceTrainer(BaseTrainer):
    def __init__(self, config, dataset) -> None:
        super().__init__(config, dataset)

        # Masking setting up. See -> gem/masks/flame for more details
        self.masking = Masking()
        mask, faces = self.load_mask("mask")
        self.mouth_mask = mask[faces]
        mask, faces = self.load_mask("neck", invert=True)
        self.tex_to_mesh = self.get_tex_to_mesh()
        self.neck_mask = mask[faces]
        self.model.gaussian_mask = mask[self.tex_to_mesh][:, 0].type(th.bool)

    def initialize(self):
        self.model = ApperanceModel(self.config, self.dataset)

        H, W = self.config.height, self.config.width
        self.bg = th.zeros([3, H, W]).cuda() if self.bg_color == "black" else th.ones([3, H, W]).cuda()
        # self.vgg_loss = VGGPerceptualLoss().cuda()
        self.vgg_loss = VGGLoss().cuda()
        self.tb_writer = SummaryWriter(log_dir=self.config.train.tb_dir)
        self.renderer = Renderer(white_background=self.bg_color == "white").cuda()
        self.renderer.resize(H, W)

        self.optimizer = instantiate(self.config.train.optimizer, params=self.model.params_group)
        self.scheduler = instantiate(self.config.train.lr_scheduler, optimizer=self.optimizer)

    def _get_flat_uv_and_mesh(self):
        topology = self.dataset.get_topology()
        v, f = self.dataset.get_canonical_mesh()
        mesh_obj = Mesh(v, f)
        geom_fn = GeometryModule(**topology, uv_size=self.model.uv_size, flip_uv=True)
        uv = geom_fn.to_uv(mesh_obj.v[None])
        flat_uv = th.flatten(uv.permute(0, 2, 3, 1), start_dim=0, end_dim=2)
        return flat_uv, mesh_obj

    def get_tex_to_mesh(self):
        flat_uv, mesh_obj = self._get_flat_uv_and_mesh()
        trimesh_mesh = to_trimesh(mesh_obj)
        uv_np = flat_uv.cpu().numpy()
        vertex_ids = trimesh_mesh.kdtree.query(uv_np)[1]
        return th.from_numpy(vertex_ids).cuda()

    def get_k_nearest(self, k, points):
        uv_np = points.cpu().numpy()
        pc = trimesh.PointCloud(uv_np)
        distances, indices = pc.kdtree.query(uv_np, k=k)
        return th.from_numpy(indices).cuda()

    def print(self):
        self.model.count_parameters()
        logger.info(f"Scheduler = {str(type(self.scheduler).__name__).ljust(20, ' ')}")
        logger.info(f"\n" + str(self.optimizer))

    def eval(self):
        self.model.eval()

    def step(self, batch):
        loss, payload, info = self.get_loss(batch)

        if loss is None:
            return

        # Recover mode to last checkpoint in RAM memory
        if th.isnan(loss):
            self.recover()
            return

        self.save_progress(payload)
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.model.get_opt_params(), 5.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        self.save(iteration=self.curr_iter)
        self.log(info)
        self.curr_iter += 1
        self.model.step(self.curr_iter)

    def inference(self, single):
        with th.no_grad():
            pkg = self.model.predict(single)
            cameras = Renderer.to_cameras(single)
            mesh_rendering = self.render_mesh(single["root_RT"], cameras, pkg.mesh, self.neck_mask)
            vis = self.summary(pkg.pred)

            keys = ["means3D", "scales", "rotation", "opacity", "colors_precomp"]
            dict = {k: pkg.pred[k].clone().detach().cpu().numpy() for k in keys}

            gt_image = single["image"]
            alpha = single["alpha"]
            gt_image = gt_image * alpha + self.bg * (1 - alpha)
            cam_id = single["cam_idx"]
            pred_image = pkg.splats.render
            pred_alpha = pkg.splats.alpha

            return AttrDict(
                {
                    "gt_image": gt_image,
                    "pred_image": pred_image,
                    "pred_alpha": pred_alpha,
                    "cam_id": cam_id,
                    "mesh_rendering": mesh_rendering,
                    "vis": vis,
                }
            ), dict

    @staticmethod
    def register(loss, info, name):
        info[name] = loss.item()
        return loss

    @staticmethod
    def pad(image, color=1):
        _, h, w = image.shape
        if w != h:
            max_wh = max(w, h)
            wp = (max_wh - w) // 2
            hp = (max_wh - h) // 2
            image = F.pad(image[None], (wp, wp, hp, hp, 0, 0), mode="constant", value=color)[0]

        return image

    def get_mesh_mask(self, single, scale=0.01):
        mesh = Mesh(single["geom_vertices"].float(), single["geom_faces"].long())
        meshes = Meshes(verts=mesh.v[None], faces=mesh.f[None]).clone()
        n = meshes.verts_normals_packed()
        mask = self.renderer.map(Renderer.to_cameras(single), (mesh.v + n * scale)[None], mesh.f[None])

        return mask

    def get_loss(self, batch):
        B = batch["image"].shape[0]
        losses = []

        for b in range(B):
            single = get_single(batch, b)
            mesh = Mesh(single["geom_vertices"].float(), single["geom_faces"].long())
            mouth_mask = self.reasterize_mask(single, mesh)
            cam_id = single["cam_idx"]

            #### PREDICTION ####

            pkg = self.model.predict(single)
            output = pkg.splats
            pred_image = output.render
            pred_alpha = output.alpha
            visibility = output.visibility_filter
            bg = output.bg_color[:, None, None]

            #### GROUND TRUTH ####

            gt_image = single["image"]
            alpha = single["alpha"]
            gt_image = gt_image * alpha + bg * (1 - alpha)
            mouth_mask = mouth_mask * 150.0 + 1.0

            #### LOSSES ####

            loss = 0
            info = {}

            # alpha_loss = l1_loss(pred_alpha, alpha) * 0.1
            # loss += self.register(alpha_loss, info, "ALPHA")

            rgb_loss = l1_loss(pred_image, gt_image, mouth_mask) * 0.4
            loss += self.register(rgb_loss, info, "L1")

            dssim_loss = (1.0 - ssim(pred_image, gt_image)) * 0.8
            loss += self.register(dssim_loss, info, "D-SSIM")

            if self.curr_iter > self.enable_vgg_from:
                c = int(self.bg_color == "white")
                vgg_loss = self.vgg_loss(self.pad(pred_image, c)[None], self.pad(gt_image, c)[None]) * 0.05
                # vgg_loss = self.vgg_loss(pred_image[None], gt_image[None]) * 0.075
                loss += self.register(vgg_loss, info, "VGG")

            if "shadow" in pkg.pred:
                shadow_reg = th.mean(th.mean(pkg.pred.shadow**2, axis=1)) * self.model.weight_shadow_reg
                loss += self.register(shadow_reg, info, "SHADOW_REG")

            pos_reg = th.mean(th.mean(pkg.pred.delta**2, axis=1)) * self.model.weight_pos_reg
            loss += self.register(pos_reg, info, "POS_REG")

            scale_reg = th.mean(th.mean(pkg.pred.scales**2, axis=1)) * self.model.weight_scale_reg
            loss += self.register(scale_reg, info, "SCALE_REG")

            self.register(loss, info, "TOTAL")

            losses.append(loss[None])

        loss = th.cat(losses).mean()

        payload = None
        if self.curr_iter % self.config.train.log_progress_n_steps == 0:
            cameras = Renderer.to_cameras(single)
            Rt = single["root_RT"]
            self.renderer.resize(self.config.height, self.config.width)
            mesh_rendering = self.render_mesh(Rt, cameras, pkg.mesh, bg_color=bg)
            vis = self.summary(pkg.pred)
            keys = ["means3D", "scales", "rotation", "opacity", "colors_precomp"]
            dict = {k: pkg.pred[k].clone().detach().cpu().numpy() for k in keys}
            payload = (gt_image, pred_image.detach().clone(), cam_id, mesh_rendering, vis, dict)

        return loss, payload, info

    def load_mask(self, name="mask", invert=False):
        folder = "flame" if self.config.dataset_name.upper() != "MULTIFACE" else "multiface"
        path = f"masks/{folder}/{name}.ply"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Mask {path} not found!")
        color_mesh = trimesh.load(path, process=False)
        color_mask = (np.array(color_mesh.visual.vertex_colors[:, 0:3]) == [255, 0, 0])[:, 0].nonzero()[0]
        color_mask = np.array(color_mask).tolist()
        v = color_mesh.vertices
        f = color_mesh.faces
        N = len(v)
        mask = th.zeros([N, 3])
        mask[color_mask, :] = 1.0
        if invert:
            mask = th.ones([N, 3])
            mask[color_mask, :] = 0.0

        return mask.cuda(), f

    def save_progress(self, payload):
        with th.no_grad():
            if payload is None:
                return

            (gt_image, pred_image, cam_id, mesh_rendering, vis_pkg, dict) = payload

            fc = (1, 1, 1) if self.bg_color == "black" else (0, 0, 0)
            C, H, W = gt_image.shape
            heapmap = th.from_numpy(compute_heatmap(gt_image, pred_image)).permute(2, 0, 1).cuda()
            n_gaussian = vis_pkg.n_gaussian

            # H = H // 2
            # canon_normals = interpolate(vis_pkg.canonical_normals, (H, H))
            # def_normals = interpolate(vis_pkg.deformed_normals, (H, H))
            # colors = interpolate(vis_pkg.colors, (H, H))
            # if vis_pkg.image is not None:
            #     image = interpolate(vis_pkg.image, (H, H))
            # else:
            #     image = th.zeros_like(colors)

            # col_0 = th.cat([(canon_normals + 1) * 0.5, colors], dim=1)
            # col_1 = th.cat([(def_normals + 1) * 0.5, image], dim=1)

            msg = "RGB (CNN)" if not self.model.use_sh else "DC (SH)"

            progress = th.cat(
                [
                    write_text(gt_image, "Ground Truth", fc),
                    write_text(pred_image, f"Pred (N={n_gaussian})", fc),
                    heapmap,
                    write_text(mesh_rendering, "Input Mesh " + "(PCA)" if self.model.use_pca_layer else "", fc),
                    # write_text(write_text(col_0, msg, set_W=667, bottom=True), f"Canonical", set_W=667),
                    # write_text(col_1, "Deformed", set_W=667),
                ],
                dim=2,
            ).detach()
            path = self.config.train.progress_dir + f"/{str(self.curr_iter).zfill(5)}_{cam_id}.png"
            tv.utils.save_image(progress, path)
            path = os.path.join(self.config.train.run_dir, "canonical") + f"/{str(self.curr_iter).zfill(5)}_{cam_id}.npy"
            if dict is not None:
                np.save(path, dict, allow_pickle=True)

    def reasterize_mask(self, pkg, mesh):
        Rt = pkg["root_RT"]
        R = Rt[:3, :3]
        T = Rt[:3, 3]
        vertices = (R @ mesh.v.float().T).T + T
        faces = mesh.f.long()[None]
        mask = self.renderer.resterize_attributes(Renderer.to_cameras(pkg), vertices[None], faces, self.mouth_mask)[0]
        output = gaussian_blur2d(mask[:, :, :, 0, :].permute(0, 3, 1, 2), (7, 7), (2.5, 2.5))
        return output[0]

    def summary(self, pred, use_activation=True):
        fn = th.sigmoid if use_activation else lambda v: v
        with th.no_grad():
            # maps = pred.gaussian_maps

            # colors = fn(maps.rgb).clone().detach()
            # delta_position = maps.position.clone().detach()

            # canon = pred.canonical_state

            # deformed_normals = calculate_tbn_uv(canon.xyz + delta_position)[:, :, :, :, 2].permute(0, 3, 1, 2)
            # canonical_normals = calculate_tbn_uv(canon.xyz)[:, :, :, :, 2].permute(0, 3, 1, 2)

            # if self.model.use_shadow:
            #     pred.image = 1.0 - th.sigmoid(maps.shadow[0].expand(3, -1, -1))

            # Flip normlas
            # deformed_normals[:, 2, ...] *= -1.0
            # canonical_normals[:, 2, ...] *= -1.0

            pkg = {
                #"colors": colors[0],
                "image": pred.image,
                #"canonical_normals": canonical_normals[0],
                #"deformed_normals": deformed_normals[0],
                "n_gaussian": pred.n_gaussian,
            }

            return AttrDict(pkg)

    def close(self):
        self.progress_bar.close()
        self.save(name="model.pth")

    def open(self):
        self.progress_bar = tqdm(range(self.curr_iter, self.max_iter + 1))
