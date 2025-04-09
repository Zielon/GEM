from glob import glob
import os
from pathlib import Path
import numpy as np
import torch as th
import torchvision as tv
from tqdm import tqdm
import trimesh

from loguru import logger
from torch import nn
from simple_knn._C import distCUDA2
from typing import NamedTuple
from gaussians.losses import l1_loss, ssim
from gaussians.renderer import render
from gaussians.utils import (
    RGB2SH,
    SH2RGB,
    BasicPointCloud,
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
)
from models.bgnet import BgNet
from models.colorcal import ColorCalibration
from utils.error import compute_heatmap
from utils.general import build_dataset, build_loader, get_single, to_device
from utils.text import write_text


class Canonical:
    def __init__(self, config) -> None:
        self.config = config
        self.canon_config = config.train.gaussians

        self.output_path = os.path.join(config.train.run_dir, "canonical")
        self.progress_path = os.path.join(self.output_path, "progress")
        Path(self.progress_path).mkdir(parents=True, exist_ok=True)
        self.checkpoints = os.path.join(self.output_path, "checkpoints")
        Path(self.checkpoints).mkdir(parents=True, exist_ok=True)
        self.ckpt_path = os.path.join(self.output_path, "model.pt")

        self.dataset = build_dataset(self.config, frame_list=[self.canon_config.frame])
        self.spatial_lr_scale = 0.5
        self.active_sh_degree = 0
        self.max_sh_degree = self.canon_config.max_sh_degree
        self._xyz = th.empty(0)
        self._features_dc = th.empty(0)
        self._features_rest = th.empty(0)
        self._scaling = th.empty(0)
        self._rotation = th.empty(0)
        self._opacity = th.empty(0)
        self.max_radii2D = th.empty(0)
        self.xyz_gradient_accum = th.empty(0)
        self.denom = th.empty(0)
        self.optimizer = None
        self.curr_iter = 1
        self.bg = None
        self.colorcal = None
        self.enable_bg_pred_after = 700

        self.setup_geometry()
        self.setup_functions()
        self.setup_net()
        self.sample_means()
        self.create_from_pcd()
        self.training_setup()
        self.restore()

        logger.info(f"Number of Gaussians {self._xyz.shape[0]}")

    def setup_geometry(self):
        frame = self.dataset[0]
        verts = th.from_numpy(frame["geom_vertices"]).cuda().float()
        faces = th.from_numpy(frame["geom_faces"]).cuda().int()

        self.mesh = (verts, faces)

    def setup_net(self):
        width = self.config.width
        height = self.config.height
        n_cam = len(self.dataset.cameras)
        cams = self.dataset.allcameras

        self.bg = BgNet(img_res=(height, width), n_views=n_cam).cuda()
        self.colorcal = ColorCalibration(cams).cuda()

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = th.exp
        self.scaling_inverse_activation = th.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = th.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = th.nn.functional.normalize

    def sample(self):
        num_pts = self.canon_config.number
        # xyz = (np.random.random((num_pts, 3)) - 0.5) * 2.0
        # xyz *= 0.05

        # mesh = trimesh.creation.icosphere(radius=0.35, subdivisions=3)
        mesh = trimesh.load("canonical.ply", process=False)
        xyz, faces = trimesh.sample.sample_surface(mesh, num_pts)
        face_normals = mesh.face_normals
        pertrub = np.random.randn(*xyz.shape) * 0.025
        dv = face_normals[faces] * pertrub
        xyz += dv

        shs = np.random.random((num_pts, 3)) / 255.0
        normals = face_normals[faces]

        self.pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=normals)

    def sample_means(self):
        if self.canon_config.sampling.upper() == "CUBE":
            self.sample()
            return

        raise NotImplementedError("Sampling not implemented")

    def create_from_pcd(self):
        fused_point_cloud = th.tensor(np.asarray(self.pcd.points)).float().cuda()
        fused_color = RGB2SH(th.tensor(np.asarray(self.pcd.colors)).float().cuda())
        features = th.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = th.clamp_min(distCUDA2(th.from_numpy(np.asarray(self.pcd.points)).float().cuda()), 0.0000001)
        scales = th.log(th.sqrt(dist2))[..., None].repeat(1, 3)
        rots = th.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * th.ones((fused_point_cloud.shape[0], 1), dtype=th.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = th.zeros((self.get_xyz.shape[0]), device="cuda")

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self):
        self.xyz_gradient_accum = th.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = th.zeros((self.get_xyz.shape[0], 1), device="cuda")

        gauss = [
            {"params": [self._xyz], "lr": self.canon_config.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._features_dc], "lr": self.canon_config.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": self.canon_config.feature_lr / 20.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": self.canon_config.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": self.canon_config.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": self.canon_config.rotation_lr, "name": "rotation"},
        ]

        mlps = [
            {"params": self.bg.parameters(), "lr": 0.0005, "name": "bg"},
            {"params": self.colorcal.parameters(), "lr": 0.001, "name": "colorcal"},
        ]

        self.optimizer = th.optim.Adam(gauss, lr=0.0, eps=1e-15)
        self.mlp_optimizer = th.optim.Adam(mlps)

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=self.canon_config.position_lr_init * self.spatial_lr_scale,
            lr_final=self.canon_config.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=self.canon_config.position_lr_delay_mult,
            max_steps=self.canon_config.position_lr_max_steps,
        )

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.mlp_optimizer.state_dict(),
            self.bg.state_dict(),
            self.colorcal.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self):
        checkpoints = sorted(glob(self.checkpoints + "/*.pth"))
        if len(checkpoints) > 0:
            path = checkpoints[-1]
        else:
            return

        (model_args, curr_iter) = th.load(path)

        logger.info(f"3DGS restored from {Path(path).stem}")

        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            mlp_opt_dict,
            bg,
            colorcal,
            self.spatial_lr_scale,
        ) = model_args

        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.training_setup()
        self.bg.load_state_dict(bg)
        self.colorcal.load_state_dict(colorcal)
        self.optimizer.load_state_dict(opt_dict)
        self.mlp_optimizer.load_state_dict(mlp_opt_dict)

        self.curr_iter = curr_iter

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return th.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = th.zeros_like(tensor)
                stored_state["exp_avg_sq"] = th.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = th.cat((stored_state["exp_avg"], th.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = th.cat(
                    (stored_state["exp_avg_sq"], th.zeros_like(extension_tensor)), dim=0
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    th.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    th.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = th.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = th.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = th.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = th.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = th.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = th.logical_and(
            selected_pts_mask, th.max(self.get_scaling, dim=1).values > self.canon_config.percent_dense * scene_extent
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = th.zeros((stds.size(0), 3), device="cuda")
        samples = th.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = th.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = th.cat((selected_pts_mask, th.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = th.where(th.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = th.logical_and(
            selected_pts_mask, th.max(self.get_scaling, dim=1).values <= self.canon_config.percent_dense * scene_extent
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = th.logical_or(th.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        th.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += th.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(th.min(self.get_opacity, th.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def get_pkg(self, single, return_bg=True):
        pkg = {
            "means3D": self.get_xyz,
            "scales": self.get_scaling,
            "rotation": self.get_rotation,
            "opacity": self.get_opacity,
            "cov": self.get_covariance(),
            "shs": self.get_features,
            "sh_degree": self.active_sh_degree,
        }

        if return_bg:
            bg = self.bg(single)
            noise = th.rand_like(bg)
            if self.curr_iter < self.enable_bg_pred_after:
                bg = noise
            pkg["bg"] = bg

        return pkg

    def save(self, iteration=None):
        ckpt = self.capture()
        if iteration is not None:
            th.save((ckpt, iteration), self.checkpoints + "/chkpnt" + str(iteration) + ".pth")
        else:
            th.save((ckpt, iteration), self.ckpt_path)

    def run(self):
        iters = self.canon_config.iterations
        progress_bar = tqdm(range(self.curr_iter, iters + 1), desc="Training progress")

        loader = build_loader(self.dataset, **self.config.train)
        train_iter = iter(loader)
        ema_loss_for_log = 0.0
        for iteration in range(self.curr_iter, iters + 1):
            self.update_learning_rate(iteration)

            if iteration % 1000 == 0:
                self.oneupSHdegree()

            try:
                batch = next(train_iter)
                batch = to_device(batch)
            except Exception as e:
                train_iter = iter(loader)

            B = batch["image"].shape[0]
            losses = []
            for b in range(B):
                single = get_single(batch, b)
                render_pkg = self.get_pkg(single)

                pkg = render(single, render_pkg)

                gt_image = single["image"]
                if self.curr_iter < self.enable_bg_pred_after:
                    gt_image = gt_image * single["alpha"] + (1 - single["alpha"]) * render_pkg["bg"]
                cam_id = single["cam_idx"]

                pred_alpha = pkg["final_T"].detach()
                pred_image = self.colorcal(pkg["render"], single["cam_idx"])
                pred_bg = render_pkg["bg"]

                Ll1 = l1_loss(pred_image, gt_image)

                lambda_dssim = self.config.train.lambda_dssim

                bg_loss = 0
                # if self.curr_iter > self.enable_bg_pred_after:
                #     bg_loss = l1_loss(pred_bg, gt_image * pred_alpha)
                rgb_loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(pred_image, gt_image))

                loss = rgb_loss + bg_loss
                losses.append(loss[None])

            loss = th.cat(losses).mean()

            loss.backward()

            # Densification
            canon = self.canon_config
            if iteration < canon.densify_until_iter:
                radii = pkg["radii"]
                visibility_filter = pkg["visibility_filter"]
                viewspace_point_tensor = pkg["viewspace_points"]

                # Keep track of max radii in image-space for pruning
                self.max_radii2D[visibility_filter] = th.max(
                    self.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                self.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > canon.densify_from_iter and iteration % canon.densification_interval == 0:
                    size_threshold = 20 if iteration > canon.opacity_reset_interval else None
                    self.densify_and_prune(canon.densify_grad_threshold, 0.005, self.spatial_lr_scale, size_threshold)

                # if iteration % canon.opacity_reset_interval == 0:
                #     self.reset_opacity()

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.mlp_optimizer.step()
            self.mlp_optimizer.zero_grad()

            self.curr_iter = iteration

            with th.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 20 == 0:
                    progress_bar.set_postfix({"N": self.get_xyz.shape[0], "Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(20)

                if iteration % 100 == 0:
                    heapmap = th.from_numpy(compute_heatmap(gt_image, pred_image)).permute(2, 0, 1).cuda()
                    progress = th.cat(
                        [
                            write_text(gt_image, "Ground truth"),
                            write_text(pred_bg, "Pred bg"),
                            write_text(pred_image * (1.0 - pred_alpha), "Pred 3DGS"),
                            write_text(pred_image, "Pred blend"),
                            heapmap,
                        ],
                        dim=2,
                    ).detach()
                    path = self.progress_path + f"/{str(iteration).zfill(5)}_{cam_id}.png"
                    tv.utils.save_image(progress, path)

                if iteration % 1000 == 0:
                    self.save(iteration)

        progress_bar.close()
        self.save()
