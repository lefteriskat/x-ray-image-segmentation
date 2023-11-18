import albumentations as A
import hydra
from matplotlib import pyplot as plt
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf
from torch import nn

from src.models.unet import UNetBlocked, DeepLabv3


class Utils:
    def __init__(self, config: DictConfig):
        self.config = config
        self.train_transforms = A.Compose(
            [
                A.ToFloat(),
                A.OneOf(
                    [
                        A.Resize(
                            width=config.data.resize_dims,
                            height=config.data.resize_dims,
                            p=0.5,
                        ),
                        A.RandomSizedCrop(
                            min_max_height=(50, 101),
                            height=config.data.resize_dims,
                            width=config.data.resize_dims,
                            p=0.25,
                        ),
                        A.PadIfNeeded(
                            min_height=config.data.resize_dims,
                            min_width=config.data.resize_dims,
                            p=0.25,
                        ),
                    ],
                    p=1,
                ),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf(
                    [
                        A.ElasticTransform(
                            alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5
                        ),
                        A.GridDistortion(p=0.5),
                        A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1),
                    ],
                    p=0.5,
                ),
                # A.CLAHE(p=0.8),
                A.RandomBrightnessContrast(p=0.6),
                A.augmentations.transforms.ColorJitter(p=0.6),
                A.RandomGamma(p=0.8),
                ToTensorV2(),
            ]
        )

        self.val_transforms = A.Compose(
            [
                A.ToFloat(),
                A.Resize(width=config.data.resize_dims, height=config.data.resize_dims),
                ToTensorV2(),
            ]
        )

        self.test_transforms = self.val_transforms

        if config.data.enable_transforms == False:
            self.train_transforms = self.test_transforms

    def get_train_transforms(self):
        return self.train_transforms

    def get_val_transforms(self):
        return self.val_transforms

    def get_test_transforms(self):
        return self.test_transforms

    def get_model(self):
        if self.config.model.name == "unet":
            return UNetBlocked(
                in_channels=self.config.model.in_channels,
                out_channels=self.config.model.out_channels,
                unet_block=self.config.model.unet_block,
            )
        elif self.config.model.name == "deeplab":
            return DeepLabv3(in_channels=self.config.model.in_channels, out_channels=self.config.model.out_channels)
        else:
            raise NotImplementedError(
                f"{self.config.model.name} model not yet supported!"
            )

    def get_optimizer(self, model):
        if self.config.training.optimizer == "adam":
            return torch.optim.Adam(model.parameters(), lr=self.config.training.lr)
        else:
            raise NotImplementedError(
                f"{self.config.training.optimizer} optimizer not yet supported!"
            )

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def plot_predictions(self, data_batch, label_batch, predictions):     
        # Convert tensors to NumPy arrays
        data_np = data_batch.cpu().numpy()
        label_np = label_batch.cpu().numpy()
        pred_np = predictions.cpu().numpy()

        print(f"Data : {data_batch.size()}")
        print(f"Label : {data_batch.size()}")
        print(f"Pred : {data_batch.size()}")

        batches_len = data_np.shape[0]
        print(f"Batches length : {batches_len}")
        pred_channels = predictions.shape[1]
        print(f"Pred channels {pred_channels}")

        fig, axs = plt.subplots(5, batches_len, figsize=(8, 10))

        # Original Data Image
        for i in range(batches_len):
            axs[i].imshow(data_np[i, 0], cmap="gray")
            axs[i].axis("off")
            axs[i].set_title("Data")

        # Original Label Image
        for i in range(batches_len):
            axs[i + batches_len].imshow(label_np[i, :, :], cmap="gray")
            axs[i + batches_len].axis("off")
            axs[i + batches_len].set_title("Label")

        # Predicted Label Images
        pred_channels = predictions.shape[1]
        for i in range(batches_len):
            for j in range(pred_channels):
                axs[i + 2 * batches_len + j].imshow(pred_np[i, j], cmap="gray")
                axs[i + 2 * batches_len + j].axis("off")
                axs[i + 2 * batches_len + j].set_title(f"Predicted class {j}")

        plt.show()