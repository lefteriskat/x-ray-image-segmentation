import logging

import hydra
import torch
from metrics import Metrics, Metrics2
from omegaconf import DictConfig
from tqdm import tqdm
from utils import Utils

from src.data.x_ray_dataset import XRayDatasetModule

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def train_model(config: DictConfig):
    # helper class to load all training parameters
    # accoring to configuration file
    utils = Utils(config)

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
        train_avg_loss = 0
        train_accuracy = 0
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
            loss = loss_func(pred, masks_batch)  # forward-pass
            loss.backward()  # backward-pass
            optimizer.step()  # update weights

            # calculate metrics to show the user
            train_avg_loss += loss.item() / len(train_loader)
            train_accuracy += Metrics.prediction_accuracy(masks_batch, pred) / len(
                train_loader
            )
            train_accuracy2 = Metrics2.get_accuracy(pred, masks_batch)

            tp_, tn_, fp_, fn_ = Metrics.get_tp_tn_fp_fn(masks_batch, pred)

            tp += tp_
            fp += fp_
            tn += tn_
            fn += fn_

        test_sens, test_spec = Metrics.get_sensitivity_specificity(tp, tn, fp, fn)
        test_dice = Metrics.get_dice_coe(masks_batch, pred)
        test_iou = Metrics.get_IoU(tp, fp, fn)

        logger.info(
            f" - Training loss: {train_avg_loss}  - Training accuracy: {train_accuracy}"
        )
        logger.info(
            f" - Training loss: {train_avg_loss}  - Training accuracy: {train_accuracy2}"
        )
        # logger.info(f" - Train sensitivity: {test_sens}  - Train specificity: {test_spec}")
        # logger.info(f" - Train DICE: {test_dice}  - Train IoU: {test_iou}")

        # Compute the evaluation set loss
        validation_avg_loss = 0
        validation_accuracy = 0
        tp, tn, fp, fn = [0] * 4

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
            validation_accuracy += Metrics.prediction_accuracy(
                masks_batch, val_pred
            ) / len(val_loader)

            tp_, tn_, fp_, fn_ = Metrics.get_tp_tn_fp_fn(masks_batch, val_pred)

            tp += tp_
            fp += fp_
            tn += tn_
            fn += fn_

        test_sens, test_spec = Metrics.get_sensitivity_specificity(tp, tn, fp, fn)
        test_dice = Metrics.get_dice_coe(masks_batch, val_pred)
        test_iou = Metrics.get_IoU(tp, fp, fn)

        logger.info(
            f" - Validation loss: {validation_avg_loss}  - Validation accuracy: {validation_accuracy}"
        )
        # logger.info(f" - Validation sensitivity: {test_sens}  - Validation specificity: {test_spec}")
        # logger.info(f" - Validation DICE: {test_dice}  - Validation IoU: {test_iou}")

        # Adjust lr
        # scheduler.step()

    # Test results and plot
    test_avg_loss = 0
    test_accuracy = 0
    tp, tn, fp, fn = [0] * 4
    for images_batch, masks_batch in tqdm(test_loader, desc="Test"):
        images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)
        with torch.no_grad():
            pred_test = model(images_batch)

        test_avg_loss += loss_func(pred_test, masks_batch) / len(test_loader)
        test_accuracy += Metrics.prediction_accuracy(masks_batch, pred_test) / len(
            test_loader
        )

        tp_, tn_, fp_, fn_ = Metrics.get_tp_tn_fp_fn(masks_batch, pred_test)

        tp += tp_
        fp += fp_
        tn += tn_
        fn += fn_

    test_sens, test_spec = Metrics.get_sensitivity_specificity(tp, tn, fp, fn)
    test_dice = Metrics.get_dice_coe(masks_batch, pred_test)
    test_iou = Metrics.get_IoU(tp, fp, fn)

    logger.info(f" - Test loss: {test_avg_loss}  - Test accuracy: {test_accuracy}")
    # logger.info(f" - Test sensitivity: {test_sens}  - Test specificity: {test_spec}")
    # logger.info(f" - Test DICE: {test_dice}  - Test IoU: {test_iou}")

    # utils.plot_predictions(images_batch, masks_batch, pred)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    train_model()
