import logging
import os

import hydra
import torch
import torch.nn.functional as F

from metrics import Metrics2
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from utils import Utils

import wandb
from src.data.x_ray_dataset import XRayDatasetModule

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def train_model(config: DictConfig):
    # helper class to load all training parameters
    # accoring to configuration file
    utils = Utils(config)

    # wandb
    wandb_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_key)
    print(config)
    run = wandb.init(
        project="xray-segmentation-project-test", config=OmegaConf.to_container(config)
    )

    torch.manual_seed(config.training.seed)

    epochs = config.training.epochs
    resized_dimensions = config.data.resize_dims

    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Runs on: {device}")

    # Datasets and data loaders
    train_transforms = utils.get_train_transforms()
    val_transforms = utils.get_val_transforms()
    test_transforms = utils.get_test_transforms()

    data_module = XRayDatasetModule(
        config, train_transforms, val_transforms, test_transforms
    )
    train_loader, val_loader, test_loader = data_module.getDataLoaders()

    # Model configuration
    model = utils.get_model()
    model.to(device)

    # Optimizer
    optimizer = utils.get_optimizer(model)

    # for now ignore
    # # lr scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[epochs // 4, epochs // 2, int(epochs * 0.75)], gamma=0.5
    # )

    # Loss function
    loss_func = utils.get_loss_function()

    # Training loop
    for epoch in tqdm(range(epochs), desc="Epoch"):
        train_avg_loss = 0.0
        train_accuracy = 0.0
        train__specificity = 0.0
        train__iou = 0.0
        train_dice = 0.0

        model.train()  # train mode
        tp, tn, fp, fn = [0] * 4
        for images_batch, masks_batch in tqdm(
            train_loader, leave=None, desc="Training"
        ):
            images_batch = images_batch.to(device)
            masks_batch = masks_batch.to(device)

            # set parameter gradients to zero
            optimizer.zero_grad()

            # forward
            pred = model(images_batch)

            # Apply softmax to get probabilities
            pred_probs = F.softmax(pred, dim=1)

            loss = loss_func(pred, masks_batch)  # forward-pass
            loss.backward()  # backward-pass
            optimizer.step()  # update weights

            # Log probability of prediction per target class to WandB
            prob_class_dict = {}
            for i in range(pred_probs.shape[1]): prob_class_dict[f"train_pred_prob_class_{i}"] = pred_probs[:, i].mean().item()

            # calculate metrics to show the user
            train_avg_loss += loss.item() / len(train_loader)

            train_accuracy += Metrics2.get_accuracy(pred, masks_batch) / len(
                train_loader
            )
            train__specificity += Metrics2.get_specificity(pred, masks_batch) / len(
                train_loader
            )
            train__iou += Metrics2.get_iou(pred, masks_batch) / len(train_loader)
            train_dice += Metrics2.get_dice_coef(pred, masks_batch) / len(train_loader)

        logger.info(
            f" - Training loss: {train_avg_loss}  - Training accuracy: {train_accuracy}"
        )
        logger.info(f" - Train specificity: {train__specificity}")
        logger.info(f" - Train DICE: {train_dice}  - Train IoU: {train__iou}")

        wandb.log(
            {
                "train_loss": train_avg_loss,
                "train_accuracy": train_accuracy,
                "train_specificity": train__specificity,
                "train_dice": train_dice,
                "train_iou": train__iou,
                "train_pred_prob_class": prob_class_dict

            }
        )

        ################################################################
        # Validation
        ################################################################

        # Compute the evaluation set loss
        validation_avg_loss = 0.0
        val_accuracy = 0.0
        val_specificity = 0.0
        val_iou = 0.0
        val_dice = 0.0

        model.eval()

        for images_batch, masks_batch in tqdm(
            val_loader, desc="Validation", leave=None
        ):
            images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)
            with torch.no_grad():
                val_pred = model(images_batch)

            # loss = loss_func(masks_batch, pred)
            loss = loss_func(val_pred, masks_batch)

            validation_avg_loss += loss / len(val_loader)

            val_accuracy += Metrics2.get_accuracy(val_pred, masks_batch) / len(
                val_loader
            )
            val_specificity += Metrics2.get_specificity(val_pred, masks_batch) / len(
                val_loader
            )
            val_iou += Metrics2.get_iou(val_pred, masks_batch) / len(val_loader)
            val_dice += Metrics2.get_dice_coef(val_pred, masks_batch) / len(val_loader)

        logger.info(
            f" - Validation loss: {validation_avg_loss}  - Validation accuracy: {val_accuracy}"
        )
        logger.info(f" - Validation specificity: {val_specificity}")
        logger.info(f" - Validation DICE: {val_dice}  - Validation IoU: {val_iou}")

        wandb.log(
            {
                "val_loss": validation_avg_loss,
                "val_accuracy": val_accuracy,
                "val_specificity": val_specificity,
                "val_dice": val_dice,
                "val_iou": val_iou,
            }
        )

        # Adjust lr
        # scheduler.step()

    ################################################################
    # Test
    ################################################################

    # Test results and plot
    test_avg_loss = 0.0
    test_accuracy = 0.0
    test_specificity = 0.0
    test_iou = 0.0
    test_dice = 0.0

    for images_batch, masks_batch in tqdm(test_loader, desc="Test"):
        images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)
        with torch.no_grad():
            pred_test = model(images_batch)

        test_avg_loss += loss_func(pred_test, masks_batch) / len(test_loader)

        test_accuracy += Metrics2.get_accuracy(pred_test, masks_batch) / len(
            test_loader
        )
        test_specificity += Metrics2.get_specificity(pred_test, masks_batch) / len(
            test_loader
        )
        test_iou += Metrics2.get_iou(pred_test, masks_batch) / len(test_loader)
        test_dice += Metrics2.get_dice_coef(pred_test, masks_batch) / len(test_loader)

    logger.info(f" - Test loss: {test_avg_loss}  - Test accuracy: {test_accuracy}")
    logger.info(f" - Test specificity: {test_specificity}")
    logger.info(f" - Test DICE: {test_dice}  - Test IoU: {test_iou}")

    # utils.plot_predictions(images_batch, masks_batch, pred)
    run.finish()

    if config.model.save_model:
        torch.save(
            model.state_dict(),
            os.path.join(config.model.save_path, utils.create_models_name() + ".pth"),
        )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    train_model()
