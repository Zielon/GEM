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
import pickle
from loguru import logger
import numpy as np
import torch
import torch.nn as nn
from .lbs import batch_rodrigues, lbs, vertices2landmarks


def to_tensor(array, dtype=torch.float32):
    if "torch.tensor" not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class FLAME(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 3D facial landmarks
    """

    def __init__(self):
        super(FLAME, self).__init__()
        logger.info("Creating the FLAME 2023 Decoder")
        src = str(Path(__file__).parent) + "/assets/"
        with open(src + "flame2023.pkl", "rb") as f:
            self.flame_model = Struct(**pickle.load(f, encoding="latin1"))
        self.NECK_IDX = 1
        self.batch_size = 1
        self.dtype = torch.float32
        self.use_face_contour = False
        self.faces = self.flame_model.f
        self.register_buffer(
            "faces_tensor",
            to_tensor(to_np(self.faces, dtype=np.int64), dtype=torch.long),
        )

        # Fixing remaining Shape betas
        # There are total 300 shape parameters to control FLAME; But one can use the first few parameters to express
        # the shape. For example 100 shape parameters are used for RingNet project
        default_shape = torch.zeros(
            [self.batch_size, 0],
            dtype=self.dtype,
            requires_grad=False,
        )
        self.register_parameter("shape_betas", nn.Parameter(default_shape, requires_grad=False))

        # Fixing remaining expression betas
        # There are total 100 shape expression parameters to control FLAME; But one can use the first few parameters to express
        # the expression. For example 50 expression parameters are used for RingNet project
        default_exp = torch.zeros(
            [self.batch_size, 0],
            dtype=self.dtype,
            requires_grad=False,
        )
        self.register_parameter("expression_betas", nn.Parameter(default_exp, requires_grad=False))

        # Eyeball and neck rotation
        default_eyball_pose = torch.zeros([self.batch_size, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter("eye_pose", nn.Parameter(default_eyball_pose, requires_grad=False))

        default_neck_pose = torch.zeros([self.batch_size, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter("neck_pose", nn.Parameter(default_neck_pose, requires_grad=False))

        # Fixing 3D translation since we use translation in the image plane

        self.use_3D_translation = True

        default_transl = torch.zeros([self.batch_size, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter("transl", nn.Parameter(default_transl, requires_grad=False))

        # The vertices of the template model
        self.register_buffer(
            "v_template",
            to_tensor(to_np(self.flame_model.v_template), dtype=self.dtype),
        )

        # The shape components
        shapedirs = self.flame_model.shapedirs
        # The shape components
        self.register_buffer("shapedirs", to_tensor(to_np(shapedirs), dtype=self.dtype))

        j_regressor = to_tensor(to_np(self.flame_model.J_regressor), dtype=self.dtype)
        self.register_buffer("J_regressor", j_regressor)

        # Pose blend shape basis
        num_pose_basis = self.flame_model.posedirs.shape[-1]
        posedirs = np.reshape(self.flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=self.dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(self.flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)

        self.register_buffer("lbs_weights", to_tensor(to_np(self.flame_model.weights), dtype=self.dtype))

        # Static and Dynamic Landmark embeddings for FLAME

        lmk_embeddings = np.load(src + "landmark_embedding.npy", allow_pickle=True, encoding="latin1")
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer("lmk_faces_idx", torch.from_numpy(lmk_embeddings["static_lmk_faces_idx"]).long())
        self.register_buffer("lmk_bary_coords", torch.from_numpy(lmk_embeddings["static_lmk_bary_coords"]).to(self.dtype))
        self.register_buffer("dynamic_lmk_faces_idx", lmk_embeddings["dynamic_lmk_faces_idx"].long())
        self.register_buffer("dynamic_lmk_bary_coords", lmk_embeddings["dynamic_lmk_bary_coords"].to(self.dtype))
        self.register_buffer("full_lmk_faces_idx", torch.from_numpy(lmk_embeddings["full_lmk_faces_idx"]).long())
        self.register_buffer("full_lmk_bary_coords", torch.from_numpy(lmk_embeddings["full_lmk_bary_coords"]).to(self.dtype))

    def _find_dynamic_lmk_idx_and_bcoords(
        self,
        vertices,
        pose,
        dynamic_lmk_faces_idx,
        dynamic_lmk_b_coords,
        neck_kin_chain,
        dtype=torch.float32,
    ):
        """
        Selects the face contour depending on the reletive position of the head
        Input:
            vertices: N X num_of_vertices X 3
            pose: N X full pose
            dynamic_lmk_faces_idx: The list of contour face indexes
            dynamic_lmk_b_coords: The list of contour barycentric weights
            neck_kin_chain: The tree to consider for the relative rotation
            dtype: Data type
        return:
            The contour face indexes and the corresponding barycentric weights
        Source: Modified for batches from https://github.com/vchoutas/smplx
        """

        batch_size = vertices.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1, neck_kin_chain)
        rot_mats = batch_rodrigues(aa_pose.view(-1, 3)).view(batch_size, -1, 3, 3)

        rel_rot_mat = torch.eye(3, device=vertices.device, dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)).to(dtype=torch.long)
        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx, 0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords, 0, y_rot_angle)

        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def forward(self, shape_params=None, expression_params=None, pose_params=None, neck_pose=None, eye_pose=None, transl=None, delta=None):
        """
        Input:
            shape_params: N X number of shape parameters
            expression_params: N X number of expression parameters
            pose_params: N X number of pose parameters
        return:
            vertices: N X V X 3
            landmarks: N X number of landmarks X 3
        """
        betas = torch.cat(
            [shape_params, self.shape_betas, expression_params, self.expression_betas],
            dim=1,
        )
        neck_pose = neck_pose if neck_pose is not None else self.neck_pose
        eye_pose = eye_pose if eye_pose is not None else self.eye_pose
        transl = transl if transl is not None else self.transl
        full_pose = torch.cat([pose_params[:, :3], neck_pose, pose_params[:, 3:], eye_pose], dim=1)
        template_vertices = self.v_template.unsqueeze(0).repeat(self.batch_size, 1, 1)

        if delta is not None:
            template_vertices += delta

        vertices, J, A, W = lbs(
            betas,
            full_pose,
            template_vertices,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
        )

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).repeat(self.batch_size, 1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(self.batch_size, 1, 1)
        if self.use_face_contour:

            (
                dyn_lmk_faces_idx,
                dyn_lmk_bary_coords,
            ) = self._find_dynamic_lmk_idx_and_bcoords(
                vertices,
                full_pose,
                self.dynamic_lmk_faces_idx,
                self.dynamic_lmk_bary_coords,
                self.neck_kin_chain,
                dtype=self.dtype,
            )

            lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks = vertices2landmarks(vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords)

        if self.use_3D_translation:
            landmarks += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        return vertices, J, A, W
