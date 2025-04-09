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


import torch
import time
import logging

from loguru import logger


class cuda_timer:
    def __init__(self, message, active):
        self.active = active
        self.message = message
        self.start = None
        self.end = None

    def __enter__(self):
        if not self.active:
            return
        torch.cuda.synchronize()
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_value, tracebac):
        if not self.active:
            return
        torch.cuda.synchronize()
        elapesd = (time.perf_counter() - self.start) * 1000.0
        logger.info(f'CUDA TIMER {elapesd:.3f}ms {self.message.upper()}')


class cpu_timer:
    def __init__(self, message, active=True):
        self.active = active
        self.message = message
        self.start = None
        self.end = None

    def __enter__(self):
        if not self.active:
            return
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_value, tracebac):
        if not self.active:
            return
        elapesd = (time.perf_counter() - self.start) * 1000.0
        logger.info(f'CPU  TIMER {elapesd:.3f}ms {self.message.upper()}')