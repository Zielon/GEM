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


from typing import Union
import torch as th
import numpy as np


class JoinDataset(th.nn.Module):
    """Combine outputs of a set of datasets."""

    def __init__(self, *args):
        super(JoinDataset, self).__init__()

        self.datasets = args

    def __getattr__(self, attr):
        for x in self.datasets:
            try:
                return x.__getattribute__(attr)
            except:
                pass

        raise AttributeError("Can't find", attr, "on", x.__class__)

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        out = {}
        for d in self.datasets:
            out.update(d[idx])
        return out


def linear2color_corr(img: Union[th.Tensor, np.ndarray], dim: int = -1) -> Union[th.Tensor, np.ndarray]:
    """Applies ad-hoc 'color correction' to a linear RGB Mugsy image along
    color channel `dim` and returns the gamma-corrected result."""

    if dim == -1:
        dim = len(img.shape) - 1

    gamma = 2.0
    black = 3.0 / 255.0
    color_scale = [1.4, 1.1, 1.6]

    assert img.shape[dim] == 3
    if dim == -1:
        dim = len(img.shape) - 1
    if isinstance(img, th.Tensor):
        scale = th.FloatTensor(color_scale).view([3 if i == dim else 1 for i in range(img.dim())])
        img = img * scale.to(img) / 1.1
        return th.clamp(
            (((1.0 / (1 - black)) * 0.95 * th.clamp(img - black, 0, 2)).pow(1.0 / gamma)) - 15.0 / 255.0,
            0,
            2,
        )
    else:
        scale = np.array(color_scale).reshape([3 if i == dim else 1 for i in range(img.ndim)])
        img = img * scale / 1.1
        return np.clip(
            (((1.0 / (1 - black)) * 0.95 * np.clip(img - black, 0, 2)) ** (1.0 / gamma)) - 15.0 / 255.0,
            0,
            2,
        )


def load_obj(filename):
    vertices = []
    faces_vertex, faces_uv = [], []
    uvs = []
    with open(filename, "r") as f:
        for s in f:
            l = s.strip()
            if len(l) == 0:
                continue
            parts = l.split(" ")
            if parts[0] == "vt":
                uvs.append([float(x) for x in parts[1:]])
            elif parts[0] == "v":
                vertices.append([float(x) for x in parts[1:]])
            elif parts[0] == "f":
                faces_vertex.append([int(x.split("/")[0]) for x in parts[1:]])
                faces_uv.append([int(x.split("/")[1]) for x in parts[1:]])
    # make sure triangle ids are 0 indexed
    obj = {
        "verts": np.array(vertices, dtype=np.float32),
        "uvs": np.array(uvs, dtype=np.float32),
        "vert_ids": np.array(faces_vertex, dtype=np.int32) - 1,
        "uv_ids": np.array(faces_uv, dtype=np.int32) - 1,
    }
    return obj


# Input is R, t in opencv spave
def opencv_to_opengl(Rt):
    Rt[[1, 2]] *= -1  # opencv to opengl coordinate system swap y,z

    """
            | R | t |
            | 0 | 1 |

            inverse is

            | R^T | -R^T * t |
            | 0   | 1        |

    """

    # Transpose rotation (row to column wise) and adjust camera position for the new rotation matrix
    Rt = np.linalg.inv(Rt)
    return Rt


def opengl_to_opencv(Rt):
    Rt = np.linalg.inv(Rt)
    Rt[[2, 1]] *= -1
    return Rt
