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


from typing import Optional
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn as nn
from sklearn.neighbors import KDTree
from loguru import logger
from kornia.filters.sobel import spatial_gradient
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.structures import Meshes
from typing import Union, Optional, Tuple
import trimesh


import json

from utils.timers import cuda_timer


class AttrDict:
    def __init__(self, entries):
        self.add_entries_(entries)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        return self.__dict__.__delitem__(key)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

    def __getattr__(self, attr):
        if attr.startswith("__"):
            return self.__getattribute__(attr)
        return self.__dict__[attr]

    def items(self):
        return self.__dict__.items()

    def __iter__(self):
        return iter(self.items())

    def add_entries_(self, entries, overwrite=True):
        for key, value in entries.items():
            if key not in self.__dict__:
                if isinstance(value, dict):
                    self.__dict__[key] = AttrDict(value)
                else:
                    self.__dict__[key] = value
            else:
                if isinstance(value, dict):
                    self.__dict__[key].add_entries_(entries=value, overwrite=overwrite)
                elif overwrite or self.__dict__[key] is None:
                    self.__dict__[key] = value

    def serialize(self):
        return json.dumps(self, default=self.obj_to_dict, indent=4)

    def obj_to_dict(self, obj):
        return obj.__dict__

    def get(self, key, default=None):
        return self.__dict__.get(key, default)


# Nx3x3
def tri2tet(tris):
    v1 = tris[:, 0, :]
    v2 = tris[:, 1, :]
    v3 = tris[:, 2, :]

    e21 = v2 - v1
    e31 = v3 - v1
    n = th.cross(e21, e31)
    n = n / th.sqrt(th.linalg.vector_norm(n, ord=2, dim=1, keepdim=True))

    v4 = v1 + n

    return v1, v2, v3, v4


def gradient(v1, v2, v3, v4):
    n = v1.shape[0]

    R = th.stack([v2 - v1, v3 - v1, v4 - v1], dim=2)
    T = v1

    RT = th.eye(4)[None].expand(n, -1, -1).cuda().float()

    RT[:, :3, :3] = R
    RT[:, :3, 3] = T

    return RT


def calculate_tbn(mesh):
    tris = mesh[0][mesh[1]]
    tets = tri2tet(tris)
    R = gradient(*tets)[:, :3, :3]
    return R


def deformation_gradient(deformed, canonical):
    canon_tris = canonical[0][canonical[1]]
    deform_tris = deformed[0][deformed[1]]
    tet_canon = tri2tet(canon_tris)
    tet_def = tri2tet(deform_tris)

    RT_canon = gradient(*tet_canon)
    RT_def = gradient(*tet_def)

    to_local = th.linalg.inv(RT_canon)
    to_deform = RT_def

    # v = canonical[0].cpu().numpy()
    # f = canonical[1].cpu().numpy()
    # mesh = trimesh.Trimesh(v, f, process=False)
    # dv = mesh.vertex_normals * 0.0
    # mesh.vertices += dv
    # xyz, faces = trimesh.sample.sample_surface(mesh, 25000)

    # new_tris = np.einsum("nik,nk->ni", R[faces].cpu().numpy(), xyz)
    # new_tris = canonical[0] + T

    return to_local, to_deform


def calucate_normal_uv(xyz):
    gradients = spatial_gradient(xyz)  # Bx3x2xHxW
    t, b = gradients[:, :, 0], gradients[:, :, 1]  # Bx3xHxW
    return F.normalize(th.cross(t, b, dim=1), dim=1, p=2)


def calculate_tbn_uv(xyz):
    gradients = spatial_gradient(xyz)  # Bx3x2xHxW
    t, b = gradients[:, :, 0], gradients[:, :, 1]  # Bx2xHxW
    normals = F.normalize(th.cross(t, b, dim=1), dim=1, p=2)
    bitangent = F.normalize(th.cross(t, normals, dim=1), dim=1, p=2)
    tangent = F.normalize(th.cross(bitangent, normals, dim=1), dim=1, p=2)

    R = th.stack([tangent.permute(0, 2, 3, 1), bitangent.permute(0, 2, 3, 1), normals.permute(0, 2, 3, 1)], dim=-1)

    return R


def deformation_uv(xyz):
    B, C, H, W = xyz.shape
    RT = th.zeros([B, H, W, 4, 4]).cuda().float()
    RT[:, :, :, 3, 3] = 1

    R = calculate_tbn_uv(xyz)
    T = xyz.clone().permute(0, 2, 3, 1)

    RT[:, :, :, :3, :3] = R
    RT[:, :, :, :3, 3] = T

    return RT


def make_uv_face_index(
    vt: th.Tensor,
    vti: th.Tensor,
    uv_shape: Union[Tuple[int, int], int],
    flip_uv: bool = True,
    device: Optional[Union[str, th.device]] = None,
):
    """Compute a UV-space face index map identifying which mesh face contains each
    texel. For texels with no assigned triangle, the index will be -1."""

    if isinstance(uv_shape, int):
        uv_shape = (uv_shape, uv_shape)

    if device is not None:
        if isinstance(device, str):
            dev = th.device(device)
        else:
            dev = device
        assert dev.type == "cuda"
    else:
        dev = th.device("cuda")

    vt = 1.0 - vt.clone()

    if flip_uv:
        vt = vt.clone()
        vt[:, 1] = 1 - vt[:, 1]
    vt_pix = 2.0 * vt.to(dev) - 1.0
    vt_pix = th.cat([vt_pix, th.ones_like(vt_pix[:, 0:1])], dim=1)
    meshes = Meshes(vt_pix[np.newaxis], vti[np.newaxis].to(dev))
    with th.no_grad():
        face_index, _, _, _ = rasterize_meshes(meshes, uv_shape, faces_per_pixel=1, z_clip_value=0.0, bin_size=0)
        face_index = face_index[0, ..., 0]
    return face_index


def make_uv_vert_index(
    vt: th.Tensor,
    vi: th.Tensor,
    vti: th.Tensor,
    uv_shape: Union[Tuple[int, int], int],
    flip_uv: bool = True,
):
    """Compute a UV-space vertex index map identifying which mesh vertices
    comprise the triangle containing each texel. For texels with no assigned
    triangle, all indices will be -1.
    """
    face_index_map = make_uv_face_index(vt, vti, uv_shape, flip_uv)
    vert_index_map = vi[face_index_map.clamp(min=0)]
    vert_index_map[face_index_map < 0] = -1
    return vert_index_map.long()


def bary_coords(points: th.Tensor, triangles: th.Tensor, eps: float = 1.0e-6):
    """Computes barycentric coordinates for a set of 2D query points given
    coordintes for the 3 vertices of the enclosing triangle for each point."""
    x = points[:, 0] - triangles[2, :, 0]
    x1 = triangles[0, :, 0] - triangles[2, :, 0]
    x2 = triangles[1, :, 0] - triangles[2, :, 0]
    y = points[:, 1] - triangles[2, :, 1]
    y1 = triangles[0, :, 1] - triangles[2, :, 1]
    y2 = triangles[1, :, 1] - triangles[2, :, 1]
    denom = y2 * x1 - y1 * x2
    n0 = y2 * x - x2 * y
    n1 = x1 * y - y1 * x

    # Small epsilon to prevent divide-by-zero error.
    denom = th.where(denom >= 0, denom.clamp(min=eps), denom.clamp(max=-eps))

    bary_0 = n0 / denom
    bary_1 = n1 / denom
    bary_2 = 1.0 - bary_0 - bary_1

    return th.stack((bary_0, bary_1, bary_2))


def make_uv_barys(
    vt: th.Tensor,
    vti: th.Tensor,
    uv_shape: Union[Tuple[int, int], int],
    flip_uv: bool = True,
):
    """Compute a UV-space barycentric map where each texel contains barycentric
    coordinates for that texel within its enclosing UV triangle. For texels
    with no assigned triangle, all 3 barycentric coordinates will be 0.
    """
    if isinstance(uv_shape, int):
        uv_shape = (uv_shape, uv_shape)

    if flip_uv:
        # Flip here because texture coordinates in some of our topo files are
        # stored in OpenGL convention with Y=0 on the bottom of the texture
        # unlike numpy/torch arrays/tensors.
        vt = vt.clone()
        vt[:, 1] = 1 - vt[:, 1]

    face_index_map = make_uv_face_index(vt, vti, uv_shape, flip_uv=False)
    vti_map = vti.long()[face_index_map.clamp(min=0)]
    uv_tri_uvs = vt[vti_map].permute(2, 0, 1, 3)

    uv_grid = th.meshgrid(
        th.linspace(0.5, uv_shape[0] - 0.5, uv_shape[0]) / uv_shape[0],
        th.linspace(0.5, uv_shape[1] - 0.5, uv_shape[1]) / uv_shape[1],
    )
    uv_grid = th.stack(uv_grid[::-1], dim=2).to(uv_tri_uvs)

    bary_map = bary_coords(uv_grid.view(-1, 2), uv_tri_uvs.view(3, -1, 2))
    bary_map = bary_map.permute(1, 0).view(uv_shape[0], uv_shape[1], 3)
    bary_map[face_index_map < 0] = 0
    return face_index_map, bary_map


def index_image_impaint(
    index_image: th.Tensor,
    bary_image: Optional[th.Tensor] = None,
    distance_threshold=100.0,
):
    # getting the mask around the indexes?
    if len(index_image.shape) == 3:
        valid_index = (index_image != -1).any(dim=-1)
    elif len(index_image.shape) == 2:
        valid_index = index_image != -1
    else:
        raise ValueError("`index_image` should be a [H,W] or [H,W,C] image")

    invalid_index = ~valid_index

    device = index_image.device

    valid_ij = th.stack(th.where(valid_index), dim=-1)
    invalid_ij = th.stack(th.where(invalid_index), dim=-1)
    lookup_valid = KDTree(valid_ij.cpu().numpy())

    dists, idxs = lookup_valid.query(invalid_ij.cpu())

    # TODO: try average?
    idxs = th.as_tensor(idxs, device=device)[..., 0]
    dists = th.as_tensor(dists, device=device)[..., 0]

    dist_mask = dists < distance_threshold

    invalid_border = th.zeros_like(invalid_index)
    invalid_border[invalid_index] = dist_mask

    invalid_src_ij = valid_ij[idxs][dist_mask]
    invalid_dst_ij = invalid_ij[dist_mask]

    index_image_imp = index_image.clone()

    index_image_imp[invalid_dst_ij[:, 0], invalid_dst_ij[:, 1]] = index_image[invalid_src_ij[:, 0], invalid_src_ij[:, 1]]

    if bary_image is not None:
        bary_image_imp = bary_image.clone()

        bary_image_imp[invalid_dst_ij[:, 0], invalid_dst_ij[:, 1]] = bary_image[invalid_src_ij[:, 0], invalid_src_ij[:, 1]]

        return index_image_imp, bary_image_imp
    return index_image_imp


class GeometryModule(nn.Module):
    def __init__(
        self,
        vi,
        vt,
        vti,
        v2uv,
        v,
        uv_size,
        flip_uv=False,
    ):
        super().__init__()

        self.register_buffer("vi", th.as_tensor(vi).cuda())
        self.register_buffer("vt", th.as_tensor(vt).cuda())
        self.register_buffer("vti", th.as_tensor(vti).cuda())
        self.register_buffer("v2uv", th.as_tensor(v2uv, dtype=th.int64))
        self.register_buffer("v", th.as_tensor(v).cuda())

        # TODO: should we just pass topology here?
        self.n_verts = v2uv.shape[0]

        self.uv_size = uv_size

        index_image = make_uv_vert_index(self.vt, self.vi, self.vti, uv_shape=uv_size, flip_uv=flip_uv)
        face_index, bary_image = make_uv_barys(self.vt, self.vti, uv_shape=uv_size, flip_uv=flip_uv)

        self.register_buffer("index_image", index_image)
        self.register_buffer("bary_image", bary_image)
        self.register_buffer("face_index_image", face_index)
        self.register_buffer("uv_faces", self.texture_to_faces(flip_uv))

    def texture_to_faces(self, flip_uv):
        uvcoords = th.cat([self.vt, self.vt[:, 0:1] * 0.0 + 1.0], -1)
        uvcoords = uvcoords * 2.0 - 1.0
        if not flip_uv:
            uvcoords[..., 1] = -uvcoords[..., 1]
        return uvcoords[self.vti]

    def render_index_images(self, uv_size, flip_uv=False, impaint=False):
        index_image = make_uv_vert_index(self.vt, self.vi, self.vti, uv_shape=uv_size, flip_uv=flip_uv)
        face_image, bary_image = make_uv_barys(self.vt, self.vti, uv_shape=uv_size, flip_uv=flip_uv)

        if impaint:
            index_image, bary_image = index_image_impaint(
                index_image,
                bary_image,
            )

        return index_image, face_image, bary_image

    def uv_mask(self):
        index_mask = th.all(self.index_image != -1, dim=-1)
        return index_mask[None]

    def vn(self, verts):
        return vert_normals(verts, self.vi[np.newaxis].to(th.long))

    def to_uv(self, values):
        return values_to_uv(values, self.index_image, self.bary_image)

    def from_uv(self, values_uv):
        # TODO: we need to sample this
        return sample_uv(values_uv, self.vt, self.v2uv.to(th.long))


def sample_uv(
    values_uv,
    uv_coords,
    v2uv: Optional[th.Tensor] = None,
    mode: str = "bilinear",
    align_corners: bool = True,
    flip_uvs: bool = False,
):
    batch_size = values_uv.shape[0]

    if flip_uvs:
        uv_coords = uv_coords.clone()
        uv_coords[:, 1] = 1.0 - uv_coords[:, 1]

    uv_coords_norm = (uv_coords * 2.0 - 1.0)[np.newaxis, :, np.newaxis].expand(batch_size, -1, -1, -1)
    values = F.grid_sample(values_uv, uv_coords_norm, align_corners=align_corners, mode=mode).squeeze(-1).permute((0, 2, 1))

    if v2uv is not None:
        values_duplicate = values[:, v2uv]
        values = values_duplicate.mean(2)

    return values


def values_to_uv(values, index_img, bary_img):
    uv_size = index_img.shape[0]
    index_mask = th.all(index_img != -1, dim=-1)
    idxs_flat = index_img[index_mask].to(th.int64)
    bary_flat = bary_img[index_mask].to(th.float32)
    # NOTE: here we assume
    values_flat = th.sum(values[:, idxs_flat].permute(0, 3, 1, 2) * bary_flat, dim=-1)
    values_uv = th.zeros(
        values.shape[0],
        values.shape[-1],
        uv_size,
        uv_size,
        dtype=values.dtype,
        device=values.device,
    )
    values_uv[:, :, index_mask] = values_flat
    return values_uv


def face_normals(v, vi, eps: float = 1e-5):
    pts = v[:, vi]
    v0 = pts[:, :, 1] - pts[:, :, 0]
    v1 = pts[:, :, 2] - pts[:, :, 0]
    n = th.cross(v0, v1, dim=-1)
    norm = th.norm(n, dim=-1, keepdim=True)
    norm[norm < eps] = 1
    n /= norm
    return n


def vert_normals(v, vi, eps: float = 1.0e-5):
    fnorms = face_normals(v, vi)
    fnorms = fnorms[:, :, None].expand(-1, -1, 3, -1).reshape(fnorms.shape[0], -1, 3)
    vi_flat = vi.view(1, -1).expand(v.shape[0], -1)
    vnorms = th.zeros_like(v)
    for j in range(3):
        vnorms[..., j].scatter_add_(1, vi_flat, fnorms[..., j])
    norm = th.norm(vnorms, dim=-1, keepdim=True)
    norm[norm < eps] = 1
    vnorms /= norm
    return vnorms


def compute_view_cos(verts, faces, camera_pos):
    vn = F.normalize(vert_normals(verts, faces), dim=-1)
    v2c = F.normalize(verts - camera_pos[:, np.newaxis], dim=-1)
    return th.einsum("bnd,bnd->bn", vn, v2c)


def compute_tbn(geom, vt, vi, vti):
    """Computes tangent, bitangent, and normal vectors given a mesh.
    Args:
        geom: [N, n_verts, 3] th.Tensor
        Vertex positions.
        vt: [n_uv_coords, 2] th.Tensor
        UV coordinates.
        vi: [..., 3] th.Tensor
        Face vertex indices.
        vti: [..., 3] th.Tensor
        Face UV indices.
    Returns:
        [..., 3] th.Tensors for T, B, N.
    """

    v0 = geom[:, vi[..., 0]]
    v1 = geom[:, vi[..., 1]]
    v2 = geom[:, vi[..., 2]]
    vt0 = vt[vti[..., 0]]
    vt1 = vt[vti[..., 1]]
    vt2 = vt[vti[..., 2]]

    v01 = v1 - v0
    v02 = v2 - v0
    vt01 = vt1 - vt0
    vt02 = vt2 - vt0
    f = 1.0 / (vt01[None, ..., 0] * vt02[None, ..., 1] - vt01[None, ..., 1] * vt02[None, ..., 0])
    tangent = f[..., None] * th.stack(
        [
            v01[..., 0] * vt02[None, ..., 1] - v02[..., 0] * vt01[None, ..., 1],
            v01[..., 1] * vt02[None, ..., 1] - v02[..., 1] * vt01[None, ..., 1],
            v01[..., 2] * vt02[None, ..., 1] - v02[..., 2] * vt01[None, ..., 1],
        ],
        dim=-1,
    )
    tangent = F.normalize(tangent, dim=-1)
    normal = F.normalize(th.cross(v01, v02, dim=3), dim=-1)
    bitangent = F.normalize(th.cross(tangent, normal, dim=3), dim=-1)

    return tangent, bitangent, normal


def compute_v2uv(vi, vti, n_max=4):
    """Computes mapping from vertex indices to texture indices.

    Args:
        vi: [F, 3], triangles
        vti: [F, 3], texture triangles
        n_max: int, max number of texture locations

    Returns:
        [n_verts, n_max], texture indices
    """
    v2uv_dict = {}
    for i_v, i_uv in zip(vi.reshape(-1), vti.reshape(-1)):
        v2uv_dict.setdefault(i_v, set()).add(i_uv)
    # assert len(v2uv_dict) == n_verts
    n_verts = vi.max() + 1
    v2uv = np.zeros((n_verts, n_max), dtype=np.int32)
    for i in v2uv_dict.keys():
        vals = sorted(list(v2uv_dict[i]))
        v2uv[i, :] = vals[0]
        v2uv[i, : len(vals)] = np.array(vals)[:n_max]
    return v2uv


def compute_neighbours(n_verts, vi, n_max_values=10):
    """Computes first-ring neighbours given vertices and faces."""
    n_vi = vi.shape[0]

    adj = {i: set() for i in range(n_verts)}
    for i in range(n_vi):
        for idx in vi[i]:
            adj[idx] |= set(vi[i]) - set([idx])

    nbs_idxs = np.tile(np.arange(n_verts)[:, np.newaxis], (1, n_max_values))
    nbs_weights = np.zeros((n_verts, n_max_values), dtype=np.float32)

    for idx in range(n_verts):
        n_values = min(len(adj[idx]), n_max_values)
        nbs_idxs[idx, :n_values] = np.array(list(adj[idx]))[:n_values]
        nbs_weights[idx, :n_values] = -1.0 / n_values

    return nbs_idxs, nbs_weights


def make_postex(v, idxim, barim):
    return (
        barim[None, :, :, 0, None] * v[:, idxim[:, :, 0]]
        + barim[None, :, :, 1, None] * v[:, idxim[:, :, 1]]
        + barim[None, :, :, 2, None] * v[:, idxim[:, :, 2]]
    ).permute(0, 3, 1, 2)


def convert_camera_parameters(Rt, K):
    R = Rt[:, :3, :3]
    t = -R.permute(0, 2, 1).bmm(Rt[:, :3, 3].unsqueeze(2)).squeeze(2)
    return dict(
        campos=t,
        camrot=R,
        focal=K[:, :2, :2],
        princpt=K[:, :2, 2],
    )


def project_points_multi(p, Rt, K, normalize=False, size=None):
    """Project a set of 3D points into multiple cameras with a pinhole model.
    Args:
        p: [B, N, 3], input 3D points in world coordinates
        Rt: [B, NC, 3, 4], extrinsics (where NC is the number of cameras to project to)
        K: [B, NC, 3, 3], intrinsics
        normalize: bool, whether to normalize coordinates to [-1.0, 1.0]
    Returns:
        tuple:
        - [B, NC, N, 2] - projected points
        - [B, NC, N] - their
    """
    B, N = p.shape[:2]
    NC = Rt.shape[1]

    Rt = Rt.reshape(B * NC, 3, 4)
    K = K.reshape(B * NC, 3, 3)

    # [B, N, 3] -> [B * NC, N, 3]
    p = p[:, np.newaxis].expand(-1, NC, -1, -1).reshape(B * NC, -1, 3)
    p_cam = p @ Rt[:, :3, :3].mT + Rt[:, :3, 3][:, np.newaxis]
    p_pix = p_cam @ K.mT
    p_depth = p_pix[:, :, 2:]
    p_pix = (p_pix[..., :2] / p_depth).reshape(B, NC, N, 2)
    p_depth = p_depth.reshape(B, NC, N)

    if normalize:
        assert size is not None
        h, w = size
        p_pix = 2.0 * p_pix / th.as_tensor([w, h], dtype=th.float32, device=p.device) - 1.0
    return p_pix, p_depth