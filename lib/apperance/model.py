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


import numpy as np
import torch as th
from gaussians.cameras import batch_to_camera
from gaussians.utils import RGB2SH, inverse_sigmoid
from lib.base_model import VOL_TRACKER_PATH, BaseModel
from lib.common import Mesh, flatten, interpolate, to_trimesh
from models.bgnet import BgNet
from models.pixcal import CameraPixelBias
import pytorch3d
import torch.nn.functional as F
from loguru import logger
from gaussians.renderer import render
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix, quaternion_multiply
from styleunet.generator import StyleUNetLight
from utils.pca_layer import PCALayer
from utils.renderer import Renderer
from models.colorcal import ColorCalibration
from simple_knn._C import distCUDA2
from utils.geometry import (
    AttrDict,
    GeometryModule,
    calculate_tbn_uv,
    calucate_normal_uv,
    deformation_uv,
)


class ApperanceModel(BaseModel):
    def __init__(self, config, dataset) -> None:
        super().__init__(config, dataset)
        self._xyz = None
        self.is_eval = False
        self.avaliable_models = ["bg", "colorcal", "generator", "pixelcal"]

        self.sampling_downscale = 2
        self.max_sh_degree = 3
        self.active_sh_degree = 0
        self.gaussian_mask = None

        v, f = self.dataset.get_canonical_mesh()
        self.canonical_mesh = Mesh(v, f)
        toplogy = self.dataset.get_topology()
        self.geom_fn = GeometryModule(**toplogy, uv_size=self.uv_size, flip_uv=True)

        self.create()

    def disable_grad(self):
        for model_name in self.avaliable_models:
            if hasattr(self, model_name):
                model = getattr(self, model_name)
                for p in model.parameters():
                    p.requires_grad_(False)

    def count_parameters(self):
        for model_name in self.avaliable_models:
            if hasattr(self, model_name):
                model = getattr(self, model_name)
                n = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"{str(type(model).__name__).ljust(20, ' ')} parameters={n}")

    def eval(self):
        self.is_eval = True
        for model_name in self.avaliable_models:
            if hasattr(self, model_name):
                model = getattr(self, model_name)
                model.eval()

    def get_opt_params(self):
        return self.generator.parameters()

    def load_state_dict(self, state):
        model_dict, colorcal_dict, pixelcal_dict, bg_dict = state

        self.generator.load_state_dict(model_dict)

        if colorcal_dict is not None:
            self.colorcal.load_state_dict(colorcal_dict)

        if pixelcal_dict is not None:
            self.pixelcal.load_state_dict(pixelcal_dict)

        if bg_dict is not None:
            self.bg.load_state_dict(bg_dict)

    def mask_gaussians(self, gaussians):
        if self.gaussian_mask is None:
            return gaussians
        output = {}
        for key in gaussians.keys():
            output[key] = gaussians[key][self.gaussian_mask]
        return output

    def state_dict(self):
        model_params = (
            self.generator.state_dict(),
            self.colorcal.state_dict() if hasattr(self, "colorcal") else None,
            self.pixelcal.state_dict() if hasattr(self, "pixelcal") else None,
            self.bg.state_dict() if hasattr(self, "bg") else None,
        )

        return model_params

    def create_map_to_mesh(self, xyz):
        mesh = to_trimesh(self.canonical_mesh)
        xyz = xyz.cpu().numpy()

        vertex_id = mesh.kdtree.query(xyz)[1]

        return th.from_numpy(vertex_id).cuda()

    def create_canonical(self):
        mesh = self.canonical_mesh
        xyz = self.geom_fn.to_uv(mesh.v[None])
        mask = ~self.geom_fn.uv_mask()

        # n = calculate_tbn_uv(xyz)[:, :, :, :, 2].permute(0, 3, 1, 2)
        # cv2.imwrite("test.png", ((n + 1) * 0.5)[0].permute(1, 2, 0).cpu().numpy() * 255)

        B, C, H, W = xyz.shape

        rots = calculate_tbn_uv(xyz)
        rots = self.rotation_activation(matrix_to_quaternion(rots)).permute(0, 3, 1, 2)

        flat_xyz = th.flatten(xyz.permute(0, 2, 3, 1), start_dim=0, end_dim=2)
        logger.info(f"Initialazing canonical with {flat_xyz.shape[0]} Gaussians")

        dist2 = th.clamp_min(distCUDA2(flat_xyz), 0.0000001)
        scales = th.log(th.sqrt(dist2))[..., None].repeat(1, 3).reshape(B, H, W, 3).permute(0, 3, 1, 2)

        opacities = inverse_sigmoid(0.7 * th.ones([B, 1, H, W]).cuda().float())

        self._xyz = xyz.contiguous()
        self._scaling = scales.contiguous()
        self._rotation = rots.contiguous()
        self._opacity = opacities.contiguous()
        self.canonical_state = AttrDict(
            {
                "xyz": self._xyz,
                "scaling": self._scaling,
                "opacity": self._opacity,
                "rotation": self._rotation,
            }
        )

        RT = deformation_uv(self._xyz)
        RT[mask] = th.eye(4).cuda(0)
        self.to_local = th.linalg.inv(RT)
        self.tex_to_mesh = self.create_map_to_mesh(flat_xyz)

    def create(self):
        self.generator = StyleUNetLight(
            self.config,
            map_channels=3,
            input_size=self.uv_size,
            output_size=self.uv_size,
            style_dim=64,
            mlp_num=4,
            channel_multiplier=self.config.train.get("unet_multiplier", 2),
        ).cuda()

        self.latents = {}
        self.create_canonical()
        self.static_latent = th.randn(1, self.generator.style_dim).cuda()

        pca = th.load(f"{VOL_TRACKER_PATH}/checkpoints/MESH_PCA/{self.config.capture_id}_mesh.ptk", weights_only=False)
        self.pca_layer = PCALayer(pca["components"], pca["mean"], pca["variance"], n_components=self.pca_n_components).cuda()

        params_group = [
            {"params": self.generator.parameters(), "lr": 0.0005, "name": "generator"},
        ]

        if self.use_bg_net:
            self.bg = BgNet(img_res=(self.config.height, self.config.width), n_views=len(self.dataset.allcameras)).cuda()
            params_group.append({"params": self.bg.parameters(), "lr": 0.0001, "name": "bg"})

        if self.use_color_calib:
            self.colorcal = ColorCalibration(self.dataset.allcameras, identity_camera=self.config.data.test_camera).cuda()
            params_group.append({"params": self.colorcal.parameters(), "lr": 0.0001, "name": "colorcal"})

        if self.use_pixel_bias:
            self.pixelcal = CameraPixelBias(
                image_height=self.config.height,
                image_width=self.config.width,
                ds_rate=8,
                cameras=self.dataset.allcameras,
            ).cuda()
            params_group.append({"params": self.pixelcal.parameters(), "lr": 0.000001, "name": "pixelcal"})

        self.params_group = params_group

    def get_style(self):
        if self.is_eval:
            return self.static_latent

        return th.randn(1, self.generator.style_dim).cuda()

    def sample_texture(self, mask, grid, tensor):
        tensor = th.flip(tensor, [2])
        sampled_image = F.grid_sample(tensor, grid, align_corners=False)

        # Test
        # sampled_meantxt = F.grid_sample(self.dataset.texmean[None], grid, align_corners=False)
        # cv2.imwrite("test.png", sampled_meantxt[0].permute(1, 2, 0).cpu().numpy()[:, :, [2, 1, 0]] * 255)

        active = mask[0][0] > 0

        extracted = sampled_image.permute(0, 2, 3, 1)[0, active, :]

        return extracted.contiguous()

    def rasterize_uv_grid(self, pkg, mesh):
        self.renderer.resize(self.config.height // self.sampling_downscale, self.config.width // self.sampling_downscale)
        cameras = Renderer.to_cameras(pkg)
        vertices = mesh.v.float()[None]
        faces = mesh.f.long()[None]

        uv = self.geom_fn.uv_faces

        uvcoords_images, mask = self.renderer.resterize_attributes(cameras, vertices, faces, uv)
        grid = uvcoords_images[:, :, :, 0, 0:2].detach()

        return {"uv_grid": grid, "visibility": mask.permute(0, 3, 1, 2).detach()}

    def sample(self, tensor, info):
        if self.use_uv_sampling:
            grid = info["uv_grid"]
            mask = info["visibility"]
            return self.sample_texture(mask, grid, tensor)
        else:
            return flatten(tensor)

    def transform_def_grad(self, grad, info):
        B, H, W, _, _ = grad.shape
        if self.use_uv_sampling:
            grid = info["uv_grid"]
            mask = info["visibility"]
            return self.sample_texture(mask, grid, grad.permute(0, 3, 4, 1, 2).reshape(B, 16, H, W)).reshape(-1, 4, 4)
        else:
            return grad.view(-1, 4, 4)

    def parse_payload(self, maps, to_deform, info, root_RT, single, to_canonical=False):

        uv_xyz = self._xyz + maps.position
        xyz = self.sample(uv_xyz, info)
        ones = th.ones(xyz.shape[0], 1).cuda().float()
        xyz_homo = th.cat([xyz, ones], dim=1)

        #### Perform deformation gradient ####

        to_deform = self.transform_def_grad(to_deform, info)
        to_local = self.transform_def_grad(self.to_local, info)

        means3D = th.einsum("nik,nk->ni", to_deform, th.einsum("nik,nk->ni", to_local, xyz_homo))[:, 0:3]

        rotation = self.sample(self.rotation_activation(self._rotation + maps.rotation), info)
        colors = self.sample(self.color_activation(maps.rgb), info)

        N = means3D.shape[0]

        #### Apply global rotation and translation ####
        if not to_canonical:
            R = root_RT[:3, :3]
            T = root_RT[:3, 3]
            means3D = (R @ means3D.T).T + T

            Q = matrix_to_quaternion(R)[None].expand(N, -1)
            rotation = quaternion_multiply(Q, rotation)

        pkg = {
            "delta": self.sample(maps.position, info),
            "means3D": means3D,
            "scales": self.sample(self.scaling_activation(self._scaling + maps.scales), info),
            "rotation": rotation,
            "opacity": self.sample(self.opacity_activation(self._opacity + maps.opacity), info),
        }

        if self.use_sh:
            fc = self.sample(maps.shs_fc, info)
            dc = RGB2SH(colors)
            shs = th.cat([dc, fc], dim=1)
            pkg["shs"] = shs.reshape(N, 16, 3)
            pkg["sh_degree"] = self.active_sh_degree
        elif self.use_shadow:
            shadow = self.sample(self.color_activation(maps.shadow), info)
            pkg["colors_precomp"] = colors * (1.0 - shadow)
            pkg["albedo"] = colors
            pkg["shadow"] = shadow
        else:
            pkg["colors_precomp"] = colors

        pkg = self.mask_gaussians(pkg)

        return pkg

    def resize(self, img, size, mode="bilinear"):
        return F.interpolate(img, size=size, mode=mode, align_corners=True)

    def calibration(self, pkg, single):
        cam_idx = th.tensor([single["cam"]]).cuda()

        pred_image = pkg["render"]
        if self.use_color_calib:
            pred_image = self.colorcal(pred_image, single["cam_idx"])

        if self.use_pixel_bias:
            pixel_bias = self.pixelcal(cam_idx)[0]
            C, H, W = pred_image.shape
            if pixel_bias.shape[1] != H:
                pixel_bias = interpolate(pixel_bias, (H, W))

            pred_image = pred_image + pixel_bias

        pkg["render"] = pred_image

        return pkg

    def compute_view_cos(self, single, uv_geom):
        camera_pos = batch_to_camera(single).camera_center
        uv_normal = calucate_normal_uv(uv_geom)
        v2c = F.normalize(uv_geom - camera_pos[None, :, None, None], dim=1)
        return th.einsum("bduv,bduv->buv", uv_normal, v2c)[:, None, ...]

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def step(self, curr_iter):
        if curr_iter % 5000 == 0:
            self.oneupSHdegree()
        self.curr_iter = curr_iter

    def predict(self, single, to_canonical=False):
        root_RT = single["root_RT"].clone()

        mesh = Mesh(single["geom_vertices"].float(), single["geom_faces"].long())
        if self.use_pca_layer:
            vertices = self.pca_layer(mesh.v)
            mesh = Mesh(vertices, mesh.f)

        uv_geom = self.geom_fn.to_uv(mesh.v[None])

        to_deform = deformation_uv(uv_geom)

        if self.use_def_grad_map:
            J = to_deform @ self.to_local
            cond_img = th.flatten(J, start_dim=3, end_dim=4).permute(0, 3, 1, 2).contiguous()
        else:
            cond_img = calucate_normal_uv(uv_geom)

        view_cond = None
        if self.use_view_cond:
            view_cond = self.compute_view_cos(single, uv_geom)

        gaussian_maps = self.generator(cond_img, self._xyz, self.get_style(), view_cond)

        info = {}
        if self.use_uv_sampling:
            info = self.rasterize_uv_grid(single, mesh)

        render_pkg = self.parse_payload(gaussian_maps, to_deform, info, root_RT, single, to_canonical)

        render_pkg["gaussian_maps"] = gaussian_maps
        render_pkg["n_gaussian"] = render_pkg["means3D"].shape[0]
        render_pkg["canonical_state"] = self.canonical_state
        # Additioanl slot for visualization
        render_pkg["image"] = None

        pkg = render(single, render_pkg, bg_color=self.bg_color, training=not self.is_eval)
        pkg = self.calibration(pkg, single)

        return AttrDict({"mesh": mesh, "splats": AttrDict(pkg), "pred": AttrDict(render_pkg)})
