# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import albumentations as A
import click
import hydra
from albumentations.pytorch import ToTensorV2
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig, OmegaConf
from x_ray_dataset import XRayDatasetModule


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # dataset = XRayDataset(input_filepath)
    # # Test dataloader : 001,002..500
    # logger.info(dataset.__len__())
    # dataset.__getitemtest__(1)
    tr = A.Compose(
        [
            A.ToFloat(),
            A.Resize(width=cfg.data.resize_dims, height=cfg.data.resize_dims),
            ToTensorV2(),
        ]
    )
    data_module = XRayDatasetModule(cfg, tr, tr, tr)
    t, _, _ = data_module.getDataLoaders()
    for images_batch, masks_batch in t:
        break


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
