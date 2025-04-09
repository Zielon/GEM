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
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import FaceLandmarkerResult
import numpy as np
import face_alignment
import torch
import torchvision.transforms.functional as Ftv
from loguru import logger
from skimage.transform import estimate_transform, warp, resize, rescale


class LiveStreamDetector:
    def __init__(self, running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM):
        base_options = python.BaseOptions(model_asset_path=f"{Path(__file__).parent.parent}/checkpoints/face_landmarker_v2_with_blendshapes.task")
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            running_mode=running_mode,
            num_faces=1,
            result_callback=self.result_callback if running_mode == mp.tasks.vision.RunningMode.LIVE_STREAM else None,
        )

        self.face_mesh = vision.FaceLandmarker.create_from_options(options)
        self.timestamp = 0
        self.image = None
        self.result = None

    def draw_landmarks(self):
        for landmark in self.landmarks:
            self.draw_circle((landmark.x, landmark.y), radius=1, color=(255, 0, 0))

    def result_callback(self, result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.result = result

    def to_image(self, camera_image):
        cv_mat = (camera_image * 255).permute(1, 2, 0).cpu().numpy()[:, :, [2, 1, 0]].astype(np.uint8)
        cv_mat = np.ascontiguousarray(cv_mat)
        self.image_height, self.image_width, _ = cv_mat.shape
        img_rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)
        return img_rgb

    def add_image(self, camera_image):
        img_rgb = self.to_image(camera_image)
        self.timestamp += 1
        self.face_mesh.detect_async(img_rgb, self.timestamp)

    def process_result(self):
        if self.result is None:
            return None, None

        if self.result.face_landmarks is None or len(self.result.face_landmarks) == 0:
            return None, None

        self.landmarks = self.result.face_landmarks[0]
        blendshapes = torch.from_numpy(np.array([face.score for face in self.result.face_blendshapes[0]])).cuda().float()

        lmks = []
        for lmk in self.landmarks:
            lmks.append(np.array([lmk.x * self.image_width, lmk.y * self.image_height]).astype(int))

        lmks = [np.array(lmks)]

        return lmks, blendshapes


class FaceDetector:
    def __init__(self, device="cuda") -> None:
        self.device = device
        self.create_detector()
        self.timestamp = 0

    def create_detector(self):
        # self.detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=self.device)
        self.detector = LiveStreamDetector(running_mode=mp.tasks.vision.RunningMode.IMAGE)
        self.use_live_stream = False

    def draw_landmarks_on_image(self, image, landmarks):
        cv_mat = (image * 255).permute(1, 2, 0).cpu().numpy()[:, :, [2, 1, 0]].astype(np.uint8)
        cv_mat = np.ascontiguousarray(cv_mat)
        for landmark in landmarks:
            x = int(landmark[0])
            y = int(landmark[1])
            cv2.circle(cv_mat, (x, y), 2, (0, 255, 0), -1)  # green circle
        return cv_mat

    def process_result(self, image):
        # C, H, W = image.shape
        # cv_mat = (image * 255).permute(1, 2, 0)[:, :, [2, 1, 0]].type(torch.uint8)
        # lmks, scores, detected_faces = self.detector.get_landmarks_from_image(cv_mat, return_landmark_score=True, return_bboxes=True)
        # if lmks is None:
        #     return None
        # return lmks

        result = self.detector.face_mesh.detect(self.detector.to_image(image))

        if len(result.face_blendshapes) == 0 or len(result.face_landmarks) == 0:
            return None, None

        blendshapes = torch.from_numpy(np.array([face.score for face in result.face_blendshapes[0]])).to(image.device).float()
        lmks = []
        for lmk in result.face_landmarks[0]:
            lmks.append(np.array([lmk.x * self.detector.image_width, lmk.y * self.detector.image_height]).astype(int))
        lmks = [np.array(lmks)]

        # annotated_image = self.draw_landmarks_on_image(image, lmks[0])
        # cv2.imwrite(f"debug/{str(self.timestamp).zfill(4)}.png", annotated_image)

        self.timestamp += 1

        return lmks, blendshapes

    def enable_live_stream(self):
        self.use_live_stream = True
        self.live_stream = LiveStreamDetector(running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM)

    @staticmethod
    def get_bbox(image, lmks, bb_scale=2.0):
        c, h, w = image.shape
        lmks = lmks[0].squeeze().astype(np.int32)
        x_min, x_max, y_min, y_max = np.min(lmks[:, 0]), np.max(lmks[:, 0]), np.min(lmks[:, 1]), np.max(lmks[:, 1])
        x_center, y_center = int((x_max + x_min) / 2.0), int((y_max + y_min) / 2.0)
        size = int(bb_scale * 2 * max(x_center - x_min, y_center - y_min))
        xb_min, xb_max, yb_min, yb_max = (
            max(x_center - size // 2, 0),
            min(x_center + size // 2, w - 1),
            max(y_center - size // 2, 0),
            min(y_center + size // 2, h - 1),
        )

        yb_max = min(yb_max, h - 1)
        xb_max = min(xb_max, w - 1)
        yb_min = max(yb_min, 0)
        xb_min = max(xb_min, 0)

        if (xb_max - xb_min) % 2 != 0:
            xb_min += 1

        if (yb_max - yb_min) % 2 != 0:
            yb_min += 1

        # top, left, height, width
        return yb_min, xb_min, yb_max, xb_max

    @staticmethod
    def crop_central(img):
        output_size = (512, 512)
        _, image_height, image_width = img.shape
        crop_height, crop_width = output_size
        if crop_width > image_width or crop_height > image_height:
            padding_ltrb = [
                (crop_width - image_width) // 2 if crop_width > image_width else 0,
                (crop_height - image_height) // 2 if crop_height > image_height else 0,
                (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
                (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
            ]
            img = Ftv.pad(img, padding_ltrb, fill=0)  # PIL uses fill value 0
            _, image_height, image_width = img.shape
            if crop_width == image_width and crop_height == image_height:
                return img

        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))

        return crop_top, crop_left, crop_height, crop_width

    @staticmethod
    def crop_image(image, bbox):
        c, h, w = image.shape
        y_min, x_min, y_max, x_max = bbox
        return image[:, max(y_min, 0) : min(y_max, h - 1), max(x_min, 0) : min(x_max, w - 1)]

    @staticmethod
    def bbox_deca(out):
        kpt = out[0].squeeze()
        left = np.min(kpt[:, 0])
        right = np.max(kpt[:, 0])
        top = np.min(kpt[:, 1])
        bottom = np.max(kpt[:, 1])
        bbox = [left, top, right, bottom]
        return bbox, "kpt68"

    @staticmethod
    def bbox2point_deca(left, right, top, bottom, type="bbox"):
        """bbox from detector and landmarks are different"""
        if type == "kpt68":
            old_size = (right - left + bottom - top) / 2 * 1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        elif type == "bbox":
            old_size = (right - left + bottom - top) / 2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.12])
        else:
            raise NotImplementedError
        return old_size, center

    @staticmethod
    def crop_image_deca(image, lmks):
        resolution_inp = 224
        bbox, bbox_type = FaceDetector.bbox_deca(lmks)
        left = bbox[0]
        right = bbox[2]
        top = bbox[1]
        bottom = bbox[3]
        old_size, center = FaceDetector.bbox2point_deca(left, right, top, bottom, type=bbox_type)
        size = int(old_size * 1.25)
        src_pts = np.array(
            [[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2], [center[0] + size / 2, center[1] - size / 2]]
        )
        DST_PTS = np.array([[0, 0], [0, resolution_inp - 1], [resolution_inp - 1, 0]])
        tform = estimate_transform("similarity", src_pts, DST_PTS)
        image = image.permute(1, 2, 0).cpu().numpy()
        dst_image = warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))

        return torch.from_numpy(dst_image).cuda().permute(2, 0, 1), (top, left, bottom, right)

    def crop_face(self, image, info=None, bb_scale=1.5):
        #if not self.use_live_stream:
        lmks, blendshape = self.process_result(image)
        #else:
        #    lmks, blendshape = self.live_stream.process_result()

        if lmks is None and blendshape is None:
            return None, None, None

        if lmks is None:
            logger.info(f"Landmarks not found for {info} using default face_alignemnt")
            lmks, blendshape = self.process_result(image)

        bbox = self.get_bbox(image, lmks, bb_scale)

        return bbox, lmks, blendshape
