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
import torchvision as tv
from pathlib import Path
import cv2
import numpy as np
from pytorch3d.ops import knn_points
import torch as th
from tqdm import tqdm
from encoder.encoder import ResnetEncoder
from gaussians.losses import VGGPerceptualLoss, l1_loss, l2_loss, ssim
from lib.apperance.model import ApperanceModel
from itertools import combinations
from loguru import logger
import torch.nn.functional as F
from  torch.optim.lr_scheduler import MultiStepLR
from lib.apperance.trainer import ApperanceTrainer
from lib.regressor.model import RegressorModel
from utils.face_detector import FaceDetector
from utils.general import build_loader, get_single, instantiate, to_device, to_tensor
from utils.geometry import AttrDict
from utils.renderer import Renderer
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as Ftv
from lib.F3DMM.masks.masking import Masking
from pytorch3d.transforms import matrix_to_quaternion, axis_angle_to_matrix, matrix_to_quaternion, quaternion_to_matrix, matrix_to_axis_angle

VOL_TRACKER_PATH = os.environ["VOL_TRACKER_PATH"]

L1_criterion = th.nn.L1Loss()

def axis_angle_to_quaternion(axis_angles):
    rot_mats = axis_angle_to_matrix(axis_angles)
    quaternions = matrix_to_quaternion(rot_mats)
    return quaternions

def quaternion_to_axis_angle(quaternions):
    rot_mats = quaternion_to_matrix(quaternions)
    axis_angles = matrix_to_axis_angle(rot_mats)
    return axis_angles

class RegressorTrainer(ApperanceTrainer):
    def __init__(self, config, dataset) -> None:
        super().__init__(config, dataset)

    def initialize(self):
        self.masking = Masking()
        self.model = RegressorModel(self.config, self.dataset)

        # Disable certain regions of GEM
        # See gem/masks/flame for available masks
        test_disable_regions = self.config.train.get("test_disable_regions", ["hair"])
        logger.info(f"Regions disabled for testing: {test_disable_regions}")
        masks = []
        for region in test_disable_regions:
            masks.append(self.load_mask(region)[0])

        # Apply the same mask which removes the neck gaussians
        neck_mask, _ = self.load_mask("neck", invert=True)
        masked_gaussians = neck_mask[self.get_tex_to_mesh()].bool()[:, 0]

        self.k_nearest = self.get_k_nearest(4, self.model.apperance.get_mean("geometry"))

        # Accumulate the mask
        mask = th.sum(th.stack(masks, dim=0), dim=0) > 0

        # Set the mask for which only mean from GEM will be used
        self.model.inactive_gem_mask = mask[self.get_tex_to_mesh()].bool()[:, 0][masked_gaussians]

        H, W = self.config.height, self.config.width
        self.bg = th.zeros([3, H, W]).cuda() if self.bg_color == "black" else th.ones([3, H, W]).cuda()
        self.vgg_loss = VGGPerceptualLoss().cuda()
        self.tb_writer = SummaryWriter(log_dir=self.config.train.tb_dir)
        self.renderer = Renderer(white_background=self.bg_color == "white").cuda()
        self.use_data_augmentation = self.config.train.get("use_data_augmentation", False)
        self.use_parts = self.config.train.get("use_parts", False)
        self.is_eval = False
        self.dataset.include_lbs = True
        self.model.refine_basis = True
        self.warmup = self.config.train.get("warmup", 40_000)
        self.current_sentence = ""

        self.build_optimizable_orientantion()

        params_basis = [
            {"params": self.model.apperance.parameters(), "lr": 0.000005, "name": "basis"},
            {"params": self.opt_orientaiton.parameters(), "lr": 0.00005, "name": "RT"},
        ]

        lr = self.config.train.get("lr", 0.001)
        params_regressor = [{"params": self.model.resnet.regressor.parameters(), "lr": lr, "name": "regressor"}]

        self.basis_optimizer = th.optim.Adam(params=params_basis)
        self.regressor_optimizer = instantiate(self.config.train.optimizer, params=params_regressor)
        self.regressor_scheduler = instantiate(self.config.train.lr_scheduler, optimizer=self.regressor_optimizer)

        # Current optimizer
        self.optimizer = self.basis_optimizer
        self.scheduler = None
        self.pred_codes = None

    def register(self, loss, info, name):
        prefix = "REFINE_" if self.model.refine_basis else ""
        info[prefix + name] = loss.item()
        return loss

    def build_optimizable_orientantion(self):
        name = "ROOT_RT"
        dst = f"{VOL_TRACKER_PATH}/checkpoints/{name}/{self.config.capture_id}.ptk"
        Path(dst).parent.mkdir(parents=True, exist_ok=True)

        if os.path.exists(dst):
            logger.info(f"ROOT RT loaded from {dst}")
            pt = th.load(dst, weights_only=False)
            self.opt_orientaiton = th.nn.ParameterDict()
            for key in pt.keys():
                self.opt_orientaiton[key] = th.zeros([4, 4])
            self.opt_orientaiton.load_state_dict(pt)
            self.opt_orientaiton.cuda()
            return

        frames = th.nn.ParameterDict().cuda()
        loader = build_loader(self.dataset, batch_size=5, num_workers=8, shuffle=False)
        for batch in tqdm(loader):
            for i in range(len(batch["cam_idx"])):
                single = get_single(to_device(batch), i)
                if single["cam_idx"] != self.config.data.test_camera:
                    continue
                frame_id = str(single["frame"].item())
                frames[frame_id] = th.nn.Parameter(single["root_RT"])

        self.opt_orientaiton = frames
        th.save(frames.state_dict(), dst)

    def eval(self):
        self.is_eval = True
        self.model.eval()
        self.model.resnet.to_jit()

    def load_state_dict(self, state):
        opt_params, model_params = state
        opt_dict_basis, opt_dict_orientaiton, opt_dict_regressor, scheduler_dict = opt_params
        self.basis_optimizer.load_state_dict(opt_dict_basis)
        self.regressor_optimizer.load_state_dict(opt_dict_regressor)
        self.regressor_scheduler.load_state_dict(scheduler_dict)
        self.model.load_state_dict(model_params)
        self.opt_orientaiton.load_state_dict(opt_dict_orientaiton)

    def state_dict(self):
        opt_params = (
            self.basis_optimizer.state_dict(),
            self.opt_orientaiton.state_dict(),
            self.regressor_optimizer.state_dict(),
            self.regressor_scheduler.state_dict(),
        )
        return (opt_params, self.model.state_dict())

    def laplacian_loss(self, name, points, idx):
        if name.lower() != "rotation":
            loss = self.geometry_laplacian_loss(points, idx)
        else:
            loss = self.rotation_laplacian_loss(points, idx)
        return loss

    def geometry_laplacian_loss(self, points, idx):
        neighbor_points = points[idx]
        mean_neighbors = neighbor_points.mean(dim=1)
        loss = th.mean((points - mean_neighbors).pow(2))
        return loss

    def rotation_laplacian_loss(self, quaternions, idx):
        axis_angles = quaternion_to_axis_angle(quaternions)  # shape: [N, 3]
        neighbor_axis_angles = axis_angles[idx]
        mean_neighbors = neighbor_axis_angles.mean(dim=1)  # shape: [N, 3]
        diff = axis_angles - mean_neighbors  # shape: [N, 3]
        loss = th.mean(diff.pow(2))
        return loss

    def inference(self, batch):
        single = get_single(batch, 0)
        with th.no_grad():
            # Each sequence/exp can jump between cameras takes
            if self.current_sentence != single["exp"]:
                self.model.reset_running_bbox()
                self.current_sentence = single["exp"]

            deca_pkg, app_pkg = self.model.predict(batch)
            cameras = Renderer.to_cameras(single)
            mesh_rendering = self.render_mesh(single["root_RT"], cameras, app_pkg[0].mesh, mask=self.neck_mask)
            vis = self.summary(app_pkg[0], use_activation=False)

            gt_image = single["image"]
            C, H, W = gt_image.shape
            if self.bg.shape != gt_image.shape:
                self.bg = th.zeros([3, H, W]).cuda() if self.bg_color == "black" else th.ones([3, H, W]).cuda()
            alpha = single["alpha"]
            if alpha != None:
                gt_image = gt_image * alpha + self.bg * (1 - alpha)
            cam_id = single["cam_idx"]
            pred_image = app_pkg[0].pred_image
            pred_alpha = app_pkg[0].pred_alpha
            deca_input = app_pkg[0].image

            return AttrDict(
                {
                    "gt_image": gt_image,
                    "pred_image": pred_image,
                    "pred_alpha": pred_alpha,
                    "deca_input": deca_input,
                    "cam_id": cam_id,
                    "mesh_rendering": mesh_rendering,
                    "vis": vis,
                }
            ), None

    def loss_weights(self, name):
        weights = {"geometry": 0.005, "rotation": 0.005, "scales": 0.005}

        if name not in weights:
            return 0.005

        return weights[name]

    def switch_to_regressor(self):
        if self.curr_iter > self.warmup and self.model.refine_basis:
            self.optimizer = self.regressor_optimizer
            self.scheduler = self.regressor_scheduler
            self.model.refine_basis = False
            logger.info("Switching to Regressor optimization. Refining Basis is done!")

    def get_loss(self, batch):
        B = batch["image"].shape[0]
        losses = []

        is_warmup = self.curr_iter < self.warmup
        self.switch_to_regressor()

        deca_pkg, app_pkg = self.model.predict(batch, is_warmup)

        if deca_pkg is None:
            return  None, None, None

        info = {}
        loss = 0.0

        #### LOSSES ####

        # Photomertric loss
        for b in range(B):
            single = get_single(batch, b)
            cam_id = single["cam_idx"]
            frame_id = str(single["frame"].item())

            #### GROUND TRUTH ####

            gt_image = single["image"]
            alpha = single["alpha"]

            # Optimizable
            single["root_RT"] = self.opt_orientaiton[frame_id]

            #### PREDICTION ####

            pred_image = app_pkg[b].pred_image
            bg = app_pkg[b].bg_color[:, None, None]
            gt_image = gt_image * alpha + bg * (1 - alpha)

            #### LOSSES ####

            color_loss = 0

            rgb_loss = l1_loss(pred_image, gt_image) * 0.3
            color_loss += self.register(rgb_loss, info, "L1")

            dssim = (1.0 - ssim(pred_image, gt_image))
            color_loss += self.register(dssim, info, "D-SSIM")

            vgg = self.vgg_loss(pred_image[None], gt_image[None]) * 0.03
            color_loss += self.register(vgg, info, "VGG")

            if self.model.refine_basis and self.curr_iter % 1000 == 0:
                self.model.apperance.make_orthagonal()

            losses.append(color_loss[None])

        loss += th.cat(losses).mean()

        ##### Regression loss #####
        if not self.model.refine_basis:
            for k in deca_pkg.pred_codes.keys():
                reg = th.mean(deca_pkg.pred_codes[k] ** 2) * 0.001
                loss += self.register(reg, info, f"{k.upper()}_REG")
                l1_codes = L1_criterion(deca_pkg.pred_codes[k], deca_pkg.gt_codes[k]) * self.loss_weights(k)
                loss += self.register(l1_codes, info, f"{k.upper()}_LOSS")

        ##### Laplacian loss #####
        if self.use_parts:
            idxs = []
            k = 4
            for pkg in app_pkg:
                points_batch = pkg.gaussian["geometry"].unsqueeze(0)
                knn_out = knn_points(points_batch, points_batch, K=k)
                idxs.append(knn_out.idx[:, :, 1:][0])
            params = [("geometry", 1000)]
            lap_loss = sum(
                (sum(self.laplacian_loss(name, pkg.gaussian[name], idx) * weight for name, weight in params) / len(params))
                for idx, pkg in zip(idxs, app_pkg)
            ) / len(app_pkg)
            # Register laplacian loss
            loss += self.register(lap_loss, info, "LAP")

        ##### Final loss #####
        self.register(loss, info, "TOTAL")

        payload = None
        if self.curr_iter % self.config.train.log_progress_n_steps == 0:
            cameras = Renderer.to_cameras(single)
            Rt = single["root_RT"]
            self.renderer.resize(self.config.height, self.config.width)
            mesh_rendering = self.render_mesh(Rt, cameras, app_pkg[-1].mesh)
            vis = self.summary(app_pkg[-1], use_activation=False)
            payload = (gt_image, pred_image.detach().clone(), cam_id, mesh_rendering, vis, None)

        return loss, payload, info
