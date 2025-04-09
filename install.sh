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

# Install required packages
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
python -m pip install torch-tensorrt tensorrt
pip install -r requirements.txt

echo -e "\nDownloading Models..."
mkdir -p checkpoints
gdown 1Wu6HInbfQdUvmsUsXk3aew4FCB41bOKI -O checkpoints/models.zip
unzip checkpoints/models.zip -d checkpoints/
rm -rf checkpoints/models.zip

# Install submodules
cd submodules/diff-gaussian-rasterization
rm -rf build
rm -rf *egg-info
pip install .

cd ../simple-knn
rm -rf build
rm -rf *egg-info
pip install .

cd ../../styleunet/stylegan2_ops
rm -rf build
rm -rf dist
rm -rf *egg-info
python3 setup.py install
