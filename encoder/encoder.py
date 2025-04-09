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
import cv2
import numpy as np
import scipy
import torch.nn as nn
import torch
import random
from loguru import logger
import torch.nn.functional as F
import torchvision.transforms.functional as Ftv
from tqdm import tqdm
from data.transfer import TransferDataset
from encoder.nvcnn import kaiming_leaky_init
from encoder.resnet import load_ResNet152Model, load_ResNet34Model, load_ResNet50Model
from encoder.smirk import SmirkEncoder
from utils.face_detector import FaceDetector
import torchvision.transforms as T
from utils.general import build_loader, copy_state_dict, get_single, to_device
from sklearn.decomposition import PCA
from utils.geometry import AttrDict
from collections import deque
from scipy.ndimage import gaussian_filter1d
from encoder.regressors import GlobalAwareAttentionMLP, AttentionMLP, SelfAttentionAttentionMLP, MLP

from math import pi, sqrt, exp

from utils.smoothing import savitzky_golay

VOL_TRACKER_PATH = os.environ["VOL_TRACKER_PATH"]


def get_gaze_from_landmarks(landmarks):
    # --- Define landmark indices based on MediaPipe Face Mesh conventions ---
    # Iris indices (these may need to be updated based on your model version):
    left_iris_indices = [469, 470, 471, 472]   # Left iris landmarks
    right_iris_indices = [474, 475, 476, 477]  # Right iris landmarks

    # Eye contour indices (approximate, common choices for eye boundaries):
    left_eye_contour = [263, 249, 390, 373, 374, 380, 381, 382, 362]
    right_eye_contour = [33, 7, 163, 144, 145, 153, 154, 155, 133]

    # --- Helper function to compute the center (mean) of a set of landmarks ---
    def compute_center(indices):
        x = sum(landmarks[i][0] for i in indices) / len(indices)
        y = sum(landmarks[i][1] for i in indices) / len(indices)
        return (x, y)

    # Compute centers for iris and eye contours for each eye:
    left_iris_center = compute_center(left_iris_indices)
    right_iris_center = compute_center(right_iris_indices)
    
    left_eye_center = compute_center(left_eye_contour)
    right_eye_center = compute_center(right_eye_contour)

    left_dx = left_iris_center[0] - left_eye_center[0]
    left_dy = left_iris_center[1] - left_eye_center[1]
    
    right_dx = right_iris_center[0] - right_eye_center[0]
    right_dy = right_iris_center[1] - right_eye_center[1]
    
    # Concatenate into a single numpy array: [left_dx, left_dy, right_dx, right_dy]
    gaze_array = np.array([left_dx, left_dy, right_dx, right_dy])

    return torch.from_numpy(gaze_array).cuda().float()


def gauss(n=11, sigma=1):
    r = range(-int(n / 2), int(n / 2) + 1)
    kernel = np.array([1 / (sigma * sqrt(2 * pi)) * exp(-float(x) ** 2 / (2 * sigma**2)) for x in r])
    kernel /= np.abs(kernel).sum()
    return kernel


class Deca(nn.Module):
    def __init__(self, output_size=236):
        super(Deca, self).__init__()
        feature_size = 2048
        self.encoder = load_ResNet50Model(False)
        # self.encoder = load_ResNet34Model(True)
        # self.encoder = load_ResNet152Model(True)
        ### regressor
        self.layers = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size),
        )

    def restore(self, checkpoint):
        copy_state_dict(self.state_dict(), checkpoint["E_flame"])

    def forward(self, inputs):
        res_features = self.encoder(inputs)
        parameters = self.layers(res_features)

        features = self.layers[0](res_features)

        return features, parameters


class ResnetEncoder(nn.Module):
    def __init__(self, outsize, config, dataset, build_pca=True):
        super(ResnetEncoder, self).__init__()
        self.deca = Deca(236).cuda()
        self.emoca = Deca(50).cuda()
        self.smirk = SmirkEncoder().cuda()
        self.config = config
        self.batch_size = config.train.batch_size
        self.bg_color = config.train.get("bg_color", "white")
        self.use_pretrained_deca = config.train.get("use_pretrained_deca", True)
        self.use_data_augmentation = config.train.get("use_data_augmentation", True)
        self.use_expr = config.train.get("use_expr", False)
        self.use_deca_relative = config.train.get("use_deca_relative", False)
        self.use_emoca_relative = config.train.get("use_emoca_relative", False)
        self.use_both_relative = config.train.get("use_both_relative", False)
        self.use_absolute = config.train.get("use_absolute", False)
        self.use_parts = config.train.get("use_parts", False)
        self.random_crop = T.RandomCrop(size=(224, 224))
        self.resize(self.config.height, self.config.width)
        self.dataset = dataset
        self.current_bbox = None
        self.canonical = None
        self.source_features = None
        self.pca_n_components_deca = config.train.get("pca_n_components_deca", 50)
        self.pca_n_components_emoca = config.train.get("pca_n_components_emoca", 50)
        self.pca_n_components = config.train.get("pca_n_components", 50)
        self.windows_size = 15
        self.sigma = 12
        self.running_window = deque(maxlen=self.windows_size)
        self.kernel = gauss(n=self.windows_size, sigma=self.sigma)
        self.is_reenactment = type(dataset) is TransferDataset

        self.augment = T.RandomChoice(
            [
                T.RandomRotation(degrees=17, fill=(0 if self.bg_color == "black" else 255)),
                T.RandomAdjustSharpness(sharpness_factor=0.5),
                T.RandomEqualize(p=0.7),
                T.ColorJitter(brightness=0.7, hue=0.3),
                T.RandomPosterize(bits=4),
                T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
            ]
        )

        # if self.use_pretrained_deca:
        self.restore_deca()
        self.restore_emoca()
        self.restore_smirk()

        # Always set DECA to eval()!!!
        self.to_jit()

        self.face_detector = FaceDetector()
        # self.deca.encoder.freezer(["layer1", "layer2", "layer3", "conv1", "bn1", "maxpool"])
        self.param_dict = {"shape": 100, "tex": 50, "exp": 50, "pose": 6, "cam": 3, "light": 27}

        # Set the training identity
        self.target_features = self.get_canonical_features(self.dataset.identity_frame)

        if build_pca:
            self.create_relative_pca()

        self.expr_cond_name = config.train.get("expr_cond_name", "smirk")

        N = self.pca_n_components_deca + ((4 + self.pca_n_components + 3) if self.use_parts else 0)
        if self.use_both_relative:
            N += self.pca_n_components_emoca
        if self.use_expr:
            if self.expr_cond_name == "smirk" or self.expr_cond_name == "emoca":
                N = 50 + 4 + 3
            elif self.expr_cond_name == "mp":
                # 52 MP blendshapes + gaze + jaw
                N = 52 + 4 + 3
            else:
                raise ValueError(f"Expression condition {self.expr_cond_name} is not implemented!")

        regressor_model = config.train.get("regressor_model", None)
        n_parts = 10 if self.use_parts else 1

        logger.info(f"Regressor model: {regressor_model} with {n_parts} n_parts")

        if regressor_model == "MLP":
            self.regressor = MLP(N, n_parts, outsize)
        elif regressor_model == "GlobalAwareAttentionMLP":
            self.regressor = GlobalAwareAttentionMLP(N, n_parts, outsize)
        elif regressor_model == "AttentionMLP":
            self.regressor = AttentionMLP(N, n_parts, outsize)
        elif regressor_model == "SelfAttentionAttentionMLP":
            self.regressor = SelfAttentionAttentionMLP(N, n_parts, outsize)
        else:
            raise ValueError(f"Regressor model {regressor_model} is not implemented!")

    def get_canonical_features(self, identity):
        logger.info(f"Regressor initalizes canonical features for {identity}")
        identity = torch.from_numpy(cv2.imread(identity)[:, :, [2, 1, 0]]).permute(2, 0, 1).cuda() / 255
        single = {"image": identity, "alpha": torch.ones_like(identity)}
        with torch.no_grad():
            image, _, mp_bs = self.parse_input(single, jitter_bbox=False)
            features_deca, deca_parameters = self.deca(image)
            features_emoca, _ = self.emoca(image)
            deca = self.decompose_code(deca_parameters)
            smirk = self.smirk(image)

            expr = smirk["expression_params"]
            jaw = smirk["jaw_params"]
            shape = deca["shape"]

        # Reset
        self.current_bbox = None
        self.running_window = deque(maxlen=self.windows_size)

        return AttrDict({"deca": features_deca, "emoca": features_emoca, "expr": expr, "jaw": jaw, "shape": shape, "mp_bs": mp_bs})

    def resize(self, H, W):
        bg = torch.ones([3, H, W]).cuda()
        self.bg = bg if self.bg_color == "white" else bg * 0

    def decompose_code(self, code):
        code_dict = {}
        start = 0
        for key in self.param_dict:
            end = start + int(self.param_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == "light":
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
        return code_dict

    def disable_deca_grad(self):
        logger.info(f"DECA's ResNet grad is disabled")
        for p in self.deca.parameters():
            p.requires_grad_(False)

    def restore_deca(self):
        path = f"{VOL_TRACKER_PATH}/checkpoints/deca_model.tar"
        checkpoint = torch.load(path, weights_only=False)
        self.deca.restore(checkpoint)
        self.deca.eval()
        logger.info(f"DECA model was restored from {path}")
        return list(checkpoint["E_flame"].keys())

    def restore_smirk(self):
        path = f"{VOL_TRACKER_PATH}/checkpoints/smirk_model.pt"
        checkpoint = torch.load(path, weights_only=False)
        checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k}

        self.smirk.load_state_dict(checkpoint_encoder)
        self.smirk.eval()

    def restore_emoca(self):
        path = f"{VOL_TRACKER_PATH}/checkpoints/emoca_model.pkt"
        checkpoint = torch.load(path, weights_only=False)
        renamed = {}
        # rename
        for key in checkpoint.keys():
            # Update the checkpoint key naming
            new_key = key.replace("deca.E_expression.", "")
            renamed[new_key] = checkpoint[key]

        self.emoca.restore({"E_flame": renamed})
        logger.info(f"EMOCA model was restored from {path}")
        return list(renamed.keys())

    def to_jit(self):
        logger.info(f"Setting RESNET into JIT and eval() mode!")
        # torch.jit.enable_onednn_fusion(True)
        sample_input = torch.rand(1, 3, 224, 224).cuda()

        # Prepare DECA
        self.deca.eval()
        traced_model = torch.jit.trace(self.deca, (sample_input))
        traced_model = torch.jit.freeze(traced_model)
        self.deca = traced_model

        # Prepare EMOCA
        self.emoca.eval()
        traced_model = torch.jit.trace(self.emoca, (sample_input))
        traced_model = torch.jit.freeze(traced_model)
        self.emoca = traced_model

    def pad(self, image, resize=False):
        _, h, w = image.shape
        value = 0 if self.bg_color == "black" else 1
        if w != h:
            max_wh = max(w, h)
            wp = (max_wh - w) // 2
            hp = (max_wh - h) // 2
            image = F.pad(image[None], (wp, wp, hp, hp, 0, 0), mode="constant", value=value)[0]

        if resize:
            _, h, w = image.shape
            return F.interpolate(image[None], (int(h * 0.75), int(w * 0.75)), mode="bilinear")[0]
        else:
            return image

    def to_deca_input(self, image, jitter_bbox=False):
        scale = 1.4
        if self.training or jitter_bbox:
            scale = random.uniform(1.2, 1.8)

        return self.face_detector.crop_face(image, bb_scale=scale)

    def create_relative_pca(self):
        name = "DECA_PCA"
        prefix = "" if not self.use_absolute else "absolute_"
        dst = f"{VOL_TRACKER_PATH}/checkpoints/{name}/{prefix}{self.config.capture_id}.ptk"
        Path(dst).parent.mkdir(parents=True, exist_ok=True)

        if os.path.exists(dst):
            logger.info(f"DECA PCA loaded from {dst}")
            self.pca = AttrDict(to_device(torch.load(dst, weights_only=False)))
            return

        logger.info(f"Bulding DECA PCA into {dst} with {len(self.dataset)} samples")
        loader = build_loader(self.dataset, batch_size=8, num_workers=8, shuffle=False, use_sampler=False)
        features_deca_dict = {}
        features_emoca_dict = {}
        expr_dict = {}
        jaw_deca_dict = {}
        mp_bs_dict = {}

        self.train(mode=False)

        for _ in range(1):
            for batch in tqdm(loader):
                for i in range(8):
                    single = get_single(to_device(batch), i)
                    if single is None:
                        break

                    # Compute PCA only for the selected camera
                    if single["cam_idx"] not in ["08"]:
                       continue

                    image, _, mp_bs = self.parse_input(single, jitter_bbox=False)

                    if image is None:
                        continue

                    with torch.no_grad():
                        features_deca, deca_parameters = self.deca(image)
                        features_emoca, _ = self.emoca(image)
                        smirk = self.smirk(image)
                        expr = smirk["expression_params"]
                        jaw = smirk["jaw_params"]

                    frame_id = str(single["frame"].item())

                    if frame_id not in features_deca_dict:
                        features_deca_dict[frame_id] = []
                        features_emoca_dict[frame_id] = []
                        expr_dict[frame_id] = []
                        jaw_deca_dict[frame_id] = []
                        mp_bs_dict[frame_id] = []

                    if self.use_absolute:
                        features_deca_dict[frame_id].append(features_deca.cpu().numpy()[0])
                        features_emoca_dict[frame_id].append(features_emoca.cpu().numpy()[0])
                        expr_dict[frame_id].append(expr.cpu().numpy()[0])
                        jaw_deca_dict[frame_id].append(jaw.cpu().numpy()[0])
                        mp_bs_dict[frame_id].append(mp_bs.cpu().numpy())
                    else:
                        features_deca_dict[frame_id].append((features_deca - self.target_features.deca).cpu().numpy()[0])
                        features_emoca_dict[frame_id].append((features_emoca - self.target_features.emoca).cpu().numpy()[0])
                        expr_dict[frame_id].append((expr - self.target_features.expr).cpu().numpy()[0])
                        jaw_deca_dict[frame_id].append((jaw - self.target_features.jaw).cpu().numpy()[0])
                        mp_bs_dict[frame_id].append((mp_bs - self.target_features.mp_bs).cpu().numpy())

        self.train(mode=True)

        checkpoint = {}
        for features, name in [(features_deca_dict, "deca"), (features_emoca_dict, "emoca"), (expr_dict, "expr"), (jaw_deca_dict, "jaw"), (mp_bs_dict, "mp_bs")]:
            averaged_features = []
            for key in features.keys():
                avg = np.average(np.array(features[key]), axis=0)
                averaged_features.append(avg)

            Mat = np.array(averaged_features)
            # Mat = gaussian_filter1d(Mat, sigma=2, axis=0)
            N, C = Mat.shape
            Mat = Mat.reshape(N, -1)
            pca = PCA(n_components=min(C, 50))
            pca.fit(Mat)

            local = {
                "components": torch.from_numpy(pca.components_),
                "variance": torch.from_numpy(pca.explained_variance_),
                "mean": torch.from_numpy(pca.mean_),
            }

            checkpoint[name] = local

        torch.save(checkpoint, dst)

        self.pca = AttrDict(to_device(checkpoint))

    def project_features(self, relative, name, n_components=30):
        std = torch.sqrt(self.pca[name].variance)[:n_components]

        X = relative - self.pca[name].mean
        coeff = torch.matmul(X, self.pca[name].components[:n_components, :].T) / std
        coeff = torch.clamp(coeff, min=-2.9, max=2.9)

        projected = torch.matmul(coeff, std[:, None] * self.pca[name].components[:n_components, :]) + self.pca[name].mean

        return projected, coeff

    def smooth_bbox(self, bbox):
        self.running_window.append(bbox)
        N = len(self.running_window)
        if N == self.windows_size:
            if len(self.kernel) != N:
                self.kernel = gauss(n=N, sigma=self.sigma)
            window = np.array(self.running_window).astype(float)
            smoothed = window.T.dot(self.kernel).astype(int)
            return smoothed.tolist()

        for _ in range(self.windows_size):
            self.running_window.append(bbox)

        return bbox

    def savitzky_bbox(self, bbox):
        self.running_window.append(bbox)
        window, order = 9, 4
        coords = []
        N = len(self.running_window)
        if N == self.windows_size:
            array = np.array(self.running_window)
            for axis in range(4):
                yhat = savitzky_golay(array[:, axis], window, order)
                coords.append(int(yhat.mean()))
            return coords

        for _ in range(self.windows_size):
            self.running_window.append(bbox)

        return bbox

    def parse_input(self, single, jitter_bbox=False):
        image = single["image"]
        alpha = single["alpha"]
        # frame_id = single["frame"]
        # cam_id = single["cam"]
        # info = f"{frame_id}_{cam_id}"
        _, H, W = image.shape
        _, bH, bW = self.bg.shape

        if H != bH or bW != W:
            self.resize(H, W)

        if alpha != None:
            image = image * alpha + self.bg * (1 - alpha)

        # Make it square
        image = self.pad(image, resize=True)

        if self.face_detector.use_live_stream:
            self.face_detector.live_stream.add_image(image)

        self.current_bbox, lmks, blendshapes = self.to_deca_input(image, jitter_bbox)

        if lmks is None:
            return None, None, None

        if not self.training:
            self.current_bbox = self.smooth_bbox(self.current_bbox)
            # self.current_bbox = self.savitzky_bbox(self.current_bbox)

        image = FaceDetector.crop_image(image, self.current_bbox)

        if self.use_data_augmentation and self.training:
            image = (image * 255).type(torch.uint8)
            image = self.augment(image) / 255.0

        image = F.interpolate(image[None], (224, 224), mode="bilinear")

        return image, lmks, blendshapes

    def to_std_scale(self, features, scale):
        return F.tanh(features) * scale

    def eval_deca(self, single):
        with torch.no_grad():
            image, _, _ = self.parse_input(single)
            _, deca_parameters = self.deca(image)
            _, expressions = self.emoca(image)

        codes = self.decompose_code(deca_parameters)
        codes["exp"] = expressions
        return codes

    def features_to_paramerers(self, x):
        with torch.no_grad():
            x = self.deca_mapper[1](x)
            x = self.deca_mapper[2](x)
        return x

    def forward(self, batch, reenactment=False, scale=2.9):
        b = batch["image"].shape[0]
        image_list = []
        codes_list = []
        coeff_deca_list = []
        coeff_emoca_list = []
        gaze_list = []
        expr_list = []
        expr_regressor_list = []
        jaw_list = []
        for b in range(b):
            single = get_single(batch, b)
            with torch.no_grad():
                image, lmks, mp_bs = self.parse_input(single)
                if lmks is None:
                    return None

                if "additional" in single:
                    image = single["additional"][None]

                features_deca, deca_parameters = self.deca(image)
                codes = self.decompose_code(deca_parameters)
                features_emoca, expr_emoca = self.emoca(image)
                smirk = self.smirk(image)

            expr_deca = codes["exp"]
            expr_smirk = smirk["expression_params"]

            ########################################

            expressions = expr_smirk
            if self.expr_cond_name == "deca":
                expressions = expr_deca
            elif self.expr_cond_name == "emoca":
                expressions = expr_emoca

            ########################################

            jaw = smirk["jaw_params"]

            image_list.append(image)

            identity = self.source_features if reenactment else self.target_features

            if not self.use_absolute:
                features_deca = features_deca - identity.deca
                features_emoca = features_emoca - identity.emoca
                expressions = expressions - identity.expr
                mp_bs = mp_bs - identity.mp_bs

            _, coeff_deca = self.project_features(features_deca[0], "deca", n_components=self.pca_n_components_deca)
            _, coeff_emoca = self.project_features(features_emoca[0], "emoca", n_components=self.pca_n_components_emoca)
            projected_expr, _ = self.project_features(expressions[0], "expr", n_components=self.pca_n_components)
            projected_jaw, _ = self.project_features(jaw[0], "jaw", n_components=self.pca_n_components)
            projected_mp_bs, _ = self.project_features(mp_bs, "mp_bs", n_components=self.pca_n_components)

            gaze = get_gaze_from_landmarks(lmks[0])

            codes["exp"] = projected_expr[None]
            codes["pose"][:, 3:] = projected_jaw[None]

            coeff_deca_list.append(coeff_deca)
            coeff_emoca_list.append(coeff_emoca)
            gaze_list.append(gaze)
            expr_list.append(torch.cat([projected_mp_bs, jaw[0]], dim=-1))
            expr_regressor_list.append(projected_expr)
            jaw_list.append(jaw[0])

            codes_list.append(codes)

            #############################################

            # TEST
            # codes["exp"] = projected_expr[None]
            # codes["pose"][:, 3:] = projected_jaw

        coeff_deca = torch.stack(coeff_deca_list)
        coeff_emoca = torch.stack(coeff_emoca_list)
        gaze = torch.stack(gaze_list)
        expr = torch.stack(expr_list)
        jaw = torch.stack(jaw_list)
        expr_regressor = torch.stack(expr_regressor_list)

        # Select regressor input
        x = None
        if self.use_deca_relative:
            x = coeff_deca
        elif self.use_emoca_relative:
            x = coeff_emoca
        elif self.use_both_relative:
            x = torch.cat([coeff_deca, coeff_emoca], dim=-1)
            if self.use_parts:
                x = torch.cat([x, gaze, jaw, expr_regressor], dim=-1)
        elif self.use_expr:
          x = torch.cat([expr, gaze], dim=-1)
        else:
            raise ValueError("None regressor option was selected!")

        coeffs = self.regressor(x)
        coeffs = self.to_std_scale(coeffs, scale)

        return AttrDict({"pca": coeffs, "deca_input": image_list, "bbox": self.current_bbox, "deca": codes_list})
