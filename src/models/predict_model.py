import logging
import os

import hydra
import torch
from metrics import Metrics2
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from utils import Utils

import wandb
from src.data.x_ray_dataset import XRayDatasetModule
from src.models.unet import DeepLabv3, UNetBlocked

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def predict_model(config: DictConfig):
    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Runs on: {device}")

    model = UNetBlocked(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        unet_block=config.model.unet_block,
    )

    path = os.path.join(
        os.path.join(config.model.save_path, utils.create_models_name() + ".pth")
    )

    model.load_state_dict(torch.load(path))
    model.eval()

    dataloader = XRayDatasetModule(config).getTestDataLoader()

    for images_batch in tqdm(dataloader, desc="Test"):
        images_batch = images_batch.to(device)
        with torch.no_grad():
            pred_test = model(images_batch)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    predict_model()
