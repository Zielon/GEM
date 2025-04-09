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
from pathlib import Path
import sys
import torch
import torch.utils.data

from loguru import logger
from omegaconf import OmegaConf
from loguru import logger
from lib.apperance.trainer import ApperanceTrainer
from lib.regressor.trainer import RegressorTrainer
from utils.general import build_dataset, build_loader, seed_everything, to_device
torch.backends.cudnn.benchmark = True


def train(config):
    dataset = build_dataset(config)
    loader = build_loader(dataset, **{**config.train, **config.data})

    logger.info(f"Training with total of {len(dataset)} frames")

    trainer_type = config.train.get("trainer", None)

    if trainer_type is None:
        trainer = ApperanceTrainer(config, dataset)
    elif trainer_type.upper() == "REGRESSOR":
        trainer = RegressorTrainer(config, dataset)
    else:
        raise RuntimeError("Selected trainer mode is not supported!")

    iterations = config.train.iterations
    iterations += config.train.get("warmup", 0)
    iteration = trainer.restore(force=True)

    train_iter = iter(loader)
    trainer.open()
    trainer.model.bg_color = "random"

    while iteration < iterations + 1:
        try:
            batch = next(train_iter)
        except Exception as e:
            logger.info(f"Iterator {str(e)}")
            train_iter = iter(loader)

        batch = to_device(batch)
        trainer.step(batch)

        iteration += 1

    trainer.close()


def folders(config):
    t = config.train
    canon = os.path.join(t.run_dir, "canonical")
    for folder in [t.run_dir, t.ckpt_dir, t.tb_dir, t.progress_dir, canon, t.results_dir]:
        logger.info(f"Creating folder {folder}")
        Path(folder).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    path = sys.argv[1]
    config = OmegaConf.load(path)

    folders(config)
    seed_everything()

    OmegaConf.save(config, f"{config.train.run_dir}/config.yml")

    train(config)
