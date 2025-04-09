#!/bin/bash

# -*- coding: utf-8 -*-
#
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

echo -e "\nDownloading Experiments..."
mkdir -p experiments
gdown 1UmZtkdVb4SFOJE4_M1RTFaIDY-1ACTdn -O _experiments.zip
unzip _experiments.zip -d .
rm -rf _experiments.zip

echo -e "\nDownloading Checkpoints..."
mkdir -p checkpoints
gdown 1-ZDshNbUfBxvg2leebhfBkiLcgR4sEL_ -O _checkpoints.zip
unzip _checkpoints.zip -d .
rm -rf _checkpoints.zip
