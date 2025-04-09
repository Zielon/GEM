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
import trimesh
from utils.geometry import AttrDict


VOL_TRACKER_PATH = os.environ["VOL_TRACKER_PATH"]


def xor(A, B):
    return (A | B) & ~(A & B)


class Masking():
    def __init__(self) -> None:
        part_masks = np.load(f"{VOL_TRACKER_PATH}/lib/F3DMM/masks/FLAME_masks.pkl", allow_pickle=True, encoding="latin1")
        masks = {}
        for k, v_mask in part_masks.items():
            masks[k] = th.tensor(v_mask, dtype=th.long)
        self._masks = AttrDict(masks)

        self._mesh_mouth_interior = self._load_mask("mouth")
        self._mesh_boundry = self._load_mask("boundry")
        self._mesh_neck = self._load_mask("neck")
        self._mesh_neck_high = self._load_mask("neck_high")
        self._mesh_jaw = self._load_mask("jaw")

        mesh_path = f"{VOL_TRACKER_PATH}/masks/flame/eyes.ply"
        if os.path.exists(mesh_path):
            mesh = trimesh.load(mesh_path, process=False)
            self.faces = np.array(mesh.faces)  # shape (F, 3) of indices
        else:
            self.faces = None

    def _template(self):
        return th.zeros((5023), dtype=bool)

    def _load_mask(self, name="mask", invert=False):
        path = f"{VOL_TRACKER_PATH}/masks/flame/{name}.ply"
        if not os.path.exists(path):
            return None
        color_mesh = trimesh.load(path, process=False)
        color_mask = (np.array(color_mesh.visual.vertex_colors[:, 0:3]) == [255, 0, 0])[:, 0].nonzero()[0]
        color_mask = np.array(color_mask).tolist()
        v = color_mesh.vertices
        f = color_mesh.faces
        N = len(v)
        mask = th.zeros([N], dtype=bool)
        mask[color_mask] = True
        if invert:
            mask = th.ones([N])
            mask[color_mask] = False

        return mask

    def eyeballs(self):
        mask = self._template()
        vals = th.cat([self._masks.right_eyeball, self._masks.left_eyeball])
        mask[vals] = True
        return mask
    
    def ears(self):
        mask = self._template()
        vals = th.cat([self._masks.right_ear, self._masks.left_ear])
        mask[vals] = True
        return mask

    def neck(self):
        return self._mesh_neck

    def eye_region(self):
        mask = self._template()
        mask[self._masks.eye_region] = True
        mask = xor(mask, (mask & self.forehead()))
        return mask

    def forehead(self):
        mask = self._template()
        mask[self._masks.forehead] = True
        return mask

    def nose(self):
        mask = self._template()
        mask[self._masks.nose] = True
        return mask
    
    def mouth(self):
        mask = self._template()
        mask[self._masks.lips] = True
        return mask | self._mesh_mouth_interior

    def face(self):
        mask = self._template()
        mask[self._masks.face] = True
        return mask

    def remaning_face(self):
        mask = ~self._template()
        mask[self._mesh_neck_high] = False
        mask[self.nose() | self.mouth() | self.eye_region() | self.forehead() | self.ears() | self.eyeballs() | self.scalp() | self.neck() | self.jaw()] = False
        return mask

    def scalp(self):
        mask = self._template()
        mask[self._masks.scalp] = True
        mask[self._mesh_neck] = False
        mask[self._mesh_neck_high] = False
        return mask

    def jaw(self):
        mask = self._template()
        mask[self._mesh_jaw] = True
        mask[self.mouth()] = False
        return mask

    def neck(self):
        mask = self._template()
        mask[self._mesh_neck_high] = True
        mask[self._mesh_neck] = False
        return mask

    def full(self):
        mask = ~self._template()
        mask[self._mesh_neck] = False
        return mask

    def _mouth_eyes_masks(self):
        mask = ~self._template()
        mask[self.eyeballs() | self._mesh_mouth_interior | self._mesh_neck] = False

        dict = {
            "eyeballs": self.eyeballs(),
            "mouth": self._mesh_mouth_interior,
            "remaning_face": mask
        }

        return dict

    def _all_masks(self):
        dict = {
            "eyeballs": self.eyeballs(),
            "ears": self.ears(),
            "eye_region": self.eye_region(),
            "forehead": self.forehead(),
            "nose": self.nose(),
            "mouth": self.mouth(),
            "remaning_face": self.remaning_face(),
            "scalp": self.scalp(),
            "neck": self.neck(),
            "jaw": self.jaw(),
        }

        return dict

    def list_masks(self, device="cuda"):
        dict = self._all_masks()
        # dict = self._mouth_eyes_masks()
    
        for k, v in dict.items():
            dict[k] = v.to(device)

        return dict

    def compute_boundary(self, mask):
        if self.faces is None:
            raise ValueError("Mesh connectivity (faces) was not loaded.")
        boundary = np.zeros_like(mask, dtype=bool)
        for face in self.faces:
            face_mask = mask[face]
            if not (np.all(face_mask) or np.all(~face_mask)):
                boundary[face] = boundary[face] | face_mask

        return boundary

    def get_all_boundaries(self, device="cuda"):
        boundaries = {}
        N = 5023
        all_masks = self._all_masks()
        for region_name, mask in all_masks.items():
            mask_np = mask.cpu().numpy().astype(bool)
            boundary_np = self.compute_boundary(mask_np)
            boundaries[region_name] = th.tensor(boundary_np, dtype=th.bool, device=device)

        return boundaries
