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

actors=("253" "460" "306")
configs=("regressor_gem")

# Optionally run mesh PCA pillow genertion
# python pca_mesh.py

for actor in "${actors[@]}"; do
    # CNN Drivable Avatar
    python train.py configs/nersemble/${actor}/default.yml
    python test.py configs/nersemble/${actor}/default.yml

    # GEM
    python pca_gauss.py configs/nersemble/${actor}/default.yml

    for config in "${configs[@]}"; do
        # Show PCA
        python pca_viewer.py configs/regressor/${actor}/${config}.yml

        # Mapper
        python train.py configs/regressor/${actor}/${config}.yml
        python test.py configs/regressor/${actor}/${config}.yml

        # Reenactment
        python test.py configs/regressor/${actor}/${config}.yml videos/01.mp4
        python test.py configs/regressor/${actor}/${config}.yml videos/02.mp4
        python test.py configs/regressor/${actor}/${config}.yml videos/03.mp4

        if [ "$actor" == "306" ]; then
            python test.py configs/regressor/${actor}/${config}.yml configs/regressor/253/${config}.yml
        else
            python test.py configs/regressor/${actor}/${config}.yml configs/regressor/306/${config}.yml
        fi
    done
done
