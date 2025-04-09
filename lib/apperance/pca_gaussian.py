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
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion

from utils.geometry import AttrDict

VOL_TRACKER_PATH = os.environ["VOL_TRACKER_PATH"]


def axis_angle_to_quaternion(axis_angles: th.Tensor) -> th.Tensor:
    rot_mats = axis_angle_to_matrix(axis_angles)
    quaternions = matrix_to_quaternion(rot_mats)
    return quaternions


def _make_orthogonal(A):
    """Assume that A is a tall matrix.

    Compute the Q factor s.t. A = QR (A may be complex) and diag(R) is real and non-negative.
    """
    X, tau = th.geqrf(A)
    Q = th.linalg.householder_product(X, tau)
    # The diagonal of X is the diagonal of R (which is always real) so we normalise by its signs
    Q *= X.diagonal(dim1=-2, dim2=-1).sgn().unsqueeze(-2)
    return Q


def _is_orthogonal(Q, eps=None):
    n, k = Q.size(-2), Q.size(-1)
    Id = th.eye(k, dtype=Q.dtype, device=Q.device)
    # A reasonable eps, but not too large
    eps = 10. * n * th.finfo(Q.dtype).eps
    return th.allclose(Q.mH @ Q, Id, atol=eps)


class PCApperance(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        uv_size = config.train.uv_size
        use_parts = config.train.get("use_parts", False)
        suffix = "_parts" if use_parts else ""
        pca = th.load(f"{VOL_TRACKER_PATH}/checkpoints/GAUSSIAN_PCA_{uv_size}/{config.capture_id}{suffix}.ptk", weights_only=False)
        gt = th.load(f"{VOL_TRACKER_PATH}/checkpoints/GAUSSIAN_PCA_{uv_size}/{config.capture_id}_coeffs{suffix}.ptk", weights_only=False)
        self.n_components = pca["config"]

        self.components = nn.ParameterDict()
        self.masks = list(pca["geometry"].keys())
        self.mods = list(pca.keys())
        self.mods.remove("colors")
        self.mods.remove("tex_map")
        self.mods.remove("config")
        self.means = {}
        self.scales = {}
        self.n = sum(self.n_components.values())

        logger.info(f"Loaded PCApperance with {self.n_components} components with {self.masks}")

        static = ["config"]

        self.channels = {}
        self.gt_coeffs = {}

        for mod, masks in pca.items():
            if mod in static:
                continue
            for k, v in masks.items():
                key = self.get_key(mod, k)

                if mod == "colors" or mod == "tex_map":
                    self.register_buffer(f"colors_{k}", pca["colors"][k].float())
                    self.register_buffer(f"tex_map_{k}", pca["tex_map"][k].int())
                    continue

                var = th.from_numpy(v["variance"]).float()
                std = th.sqrt(var)
                self.gt_coeffs[key] = (th.from_numpy(gt[mod][k][:, : self.n_components[mod]]).float() / std[: self.n_components[mod]]).cuda()
                self.components[key] = th.from_numpy(v["components"]).float()[: self.n_components[mod], :]
                self.scales[key] = std[: self.n_components[mod], None].float().cuda()
                self.means[key] = th.from_numpy(v["mean"]).float().cuda()
                self.channels[key] = v["channels"]

    def get_mean(self, mod):
        mean = []
        for mask in self.masks:
            key = self.get_key(mod, mask)
            C = self.channels[key]
            mean.append(self.means[key].reshape(-1, C))
        return th.cat(mean).contiguous()

    def get_key(self, mod, mask):
        return f"{mod}_{mask}"

    def to_key_parts(self, key):
        # For backward compatibility, could have been "+" instead of "_"
        mod = key.split("_")[0]
        mask = "_".join(key.split("_")[1:])
        return mod, mask

    def resize(self, n):
        for k, v in self.components.items():
            self.gt_coeffs[k] = self.gt_coeffs[k][:, :n]
            self.components[k] = self.components[k][:n, :]
            self.scales[k] = self.scales[k][:n, :]

    def keys(self):
        return sorted(self.mods)

    def get(self, idx):
        results = {}
        for k in self.gt_coeffs.keys():
            results[k] = self.gt_coeffs[k][idx]

        return results

    def make_orthagonal(self):
        for k in self.components.keys():
            A = self.components[k].T
            if not _is_orthogonal(A):
                logger.info(f"Orthagonalizing basis of {k}")
                self.components[k] = _make_orthogonal(A).T

    def to_coeffs(self, values, is_batch=False):
        coeffs = {}
        for mask in self.masks:
            n = 0
            for mod in self.mods:
                key = self.get_key(mod, mask)
                i = self.n_components[mod]
                if is_batch:
                    coeffs[key] = values[mask][:, n:n+i]
                else:
                    coeffs[key] = values[mask][n:n+i]
                n += i

        return coeffs

    def inverse_transform(self, coeffs, masking=None):
        results = {m: [] for m in self.mods}
        results["apperance"] = []
        results["tex_map"] = []

        for k in coeffs.keys():
            C = self.channels[k]
            values = (th.matmul(coeffs[k], self.scales[k] * self.components[k]) + self.means[k]).reshape(-1, C)
            mod, name_mask = self.to_key_parts(k)

            if masking != None:
                if name_mask in masking["keys"]:
                    values = self.means[k].reshape(-1, C)
                if "mask" in masking and len(self.masks) == 1:
                    values[masking["mask"]] = self.means[k].reshape(-1, C)[masking["mask"]]

            results[mod].append(values)

        for m in self.masks:
            results["apperance"].append(getattr(self, f"colors_{m}"))
            results["tex_map"].append(getattr(self, f"tex_map_{m}"))

        results = {k: th.cat(v).contiguous() for k, v in results.items()}

        # results["rotation"] = axis_angle_to_quaternion(results["rotation"])

        return AttrDict(results)
