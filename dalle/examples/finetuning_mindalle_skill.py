# ------------------------------------------------------------------------------------
# Minimal DALL-E
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import sys
import argparse
from typing import Optional
from datetime import datetime

from pathlib import Path
import json
import PIL

import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import rank_zero_only

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dalle.models import Dalle, DalleTrain
from dalle.models import ImageGPT

class ImageLogger(Callback):
    def __init__(self):
        super().__init__()

    @rank_zero_only
    def log_img(self, pl_module, batch, current_epoch, split="train"):
        with torch.no_grad():
            images, labels = batch
            recons = pl_module.stage1(images)
            images = images.cpu()
            recons = recons.cpu()

            grid_org = (torchvision.utils.make_grid(images, nrow=8) + 1.0) / 2.0
            grid_rec = (torchvision.utils.make_grid(recons, nrow=8) + 1.0) / 2.0
            grid_rec = torch.clip(grid_rec, min=0, max=1)

            pl_module.logger.experiment.add_image(f"images_org/{split}", grid_org, global_step=current_epoch)
            pl_module.logger.experiment.add_image(f"images_rec/{split}", grid_rec, global_step=current_epoch)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0 and trainer.current_epoch < 5:
            self.log_img(pl_module, batch, current_epoch=trainer.current_epoch, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0 and trainer.current_epoch < 5:
            self.log_img(pl_module, batch, current_epoch=trainer.current_epoch, split="test")

from skill_dataset import SkillTextImageDataset

class SkillDataModule(pl.LightningDataModule):
    def __init__(self,
                dataset_dir='/home/nlplab/hdd2/yoon/juntae/DallEval/models/mindalle/minDALL-E/',
                skill='data',
                tokenizer=None,
                 image_resolution: int = 256,
                 train_batch_size: int = 1,
                 valid_batch_size: int = 32,
                 num_workers: int = 8):
        super().__init__()

        dataset_dir = Path(dataset_dir).resolve()


        train_image_dir = dataset_dir.joinpath(f'{skill}/images')
        train_text_data_file = dataset_dir.joinpath(f'{skill}/label.json')
        #valid_image_dir = dataset_dir.joinpath(f'{skill}/images')
        #valid_text_data_file = dataset_dir.joinpath(f'{skill}/label.json')

        self.train_image_dir = train_image_dir
        self.train_text_data_file = train_text_data_file

        #self.valid_image_dir = valid_image_dir
        #self.valid_text_data_file = valid_text_data_file

        self.image_resolution = image_resolution
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers

        self.tokenizer = tokenizer

    def setup(self, stage=None):

        self.trainset = SkillTextImageDataset(
            image_dir=self.train_image_dir,
            text_data_file=self.train_text_data_file,
            # transform=self.image_transform,
            image_resolution=self.image_resolution,
            tokenizer=self.tokenizer)
        # self.validset = SkillTextImageDataset(
        #     image_dir=self.valid_image_dir,
        #     text_data_file=self.valid_text_data_file,
        #     # transform=self.image_transform,
        #     image_resolution=self.image_resolution,
        #     tokenizer=self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.trainset,
                          batch_size=self.train_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)

    # def valid_dataloader(self):
    #     return DataLoader(self.validset,
    #                       batch_size=self.valid_batch_size,
    #                       num_workers=self.num_workers,
    #                       pin_memory=True)


def setup_callbacks(config):
    # Setup callbacks
    now = datetime.now().strftime('%d%m%Y_%H%M%S')
    result_path = os.path.join(args.result_path,
                            #    os.path.basename(args.config_downstream).split('.')[0],
                            #    now
                                f"{args.skill_name}_{now}"
                               )
    ckpt_path = os.path.join(result_path, 'ckpt')
    log_path = os.path.join(result_path, 'log')


    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename=f"{args.skill_name}" + "-{epoch:02d}",
        every_n_epochs=config.experiment.save_ckpt_freq,
        save_weights_only=True,
        save_last=True
    )
    logger = TensorBoardLogger(log_path, name="mindalle-skill")
    # logger_img = ImageLogger()
    # return checkpoint_callback, logger, logger_img
    return checkpoint_callback, logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--config-downstream', type=str, default=None, required=True)
    # parser.add_argument('-u', '--path-upstream', type=str, default=None, required=True)
    # parser.add_argument('-r', '--result-path', type=str, default=None, required=True)
    # parser.add_argument('--imagenet-path', type=str, default=None, required=True)

    parser.add_argument('--result_path', type=str, default=None, required=True)

    parser.add_argument('--dataset_dir', type=str, default=None, required=True)
    parser.add_argument('--skill_name', type=str, default=None, required=True)

    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)


    args = parser.parse_args()


    print(args)
    pl.seed_everything(args.seed)

    # Build iGPT
    # model, config = ImageGPT.from_pretrained(args.path_upstream, args.config_downstream)

    # model = DalleTrain.from_pretrained('minDALL-E/1.3B')  # This will automatically download the pretrained model.
    model = DalleTrain.from_pretrained(
        'minDALL-E/1.3B',
        ft_config_path=args.config_downstream,
        )  # This will automatically download the pretrained model.

    tokenizer = model.tokenizer


    config = model.config
    # Setup callbacks
    # ckpt_callback, logger, logger_img = setup_callbacks(config)
    ckpt_callback, logger = setup_callbacks(config)

    # Build data modules

    dataset = SkillDataModule(
        dataset_dir = args.dataset_dir,
        skill=args.skill_name,
        tokenizer = tokenizer,
        image_resolution=config.dataset.image_resolution,
        train_batch_size=config.experiment.local_batch_size,
        valid_batch_size=config.experiment.valid_batch_size,
        num_workers=4)


    dataset.setup()
    train_dataloader = dataset.train_dataloader()
    #valid_dataloader = dataset.valid_dataloader()
    print(f"len(train_dataset) = {len(dataset.trainset)}")
    #print(f"len(valid_dataset) = {len(dataset.validset)}")

    # Calculate how many batches are accumulated
    assert config.experiment.total_batch_size % (config.experiment.local_batch_size * args.n_gpus) == 0
    grad_accm_steps = config.experiment.total_batch_size // (config.experiment.local_batch_size * args.n_gpus)
    config.optimizer.max_steps = len(dataset.trainset) // config.experiment.total_batch_size * config.experiment.epochs

    # Build trainer
    trainer = pl.Trainer(max_epochs=config.experiment.epochs,
                         accumulate_grad_batches=grad_accm_steps,
                         gradient_clip_val=config.optimizer.grad_clip_norm,
                         precision=16 if config.experiment.use_amp else 32,
                        #  callbacks=[ckpt_callback, logger_img],
                          callbacks=[ckpt_callback],
                         accelerator="gpu",
                         devices=args.n_gpus,
                         strategy="ddp",
                         logger=logger)
    #trainer.fit(model, train_dataloader, valid_dataloader)
    trainer.fit(model, train_dataloader)
