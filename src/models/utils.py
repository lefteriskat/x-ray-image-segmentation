import albumentations as A
import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from matplotlib.gridspec import GridSpec
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from src.models.unet import DeepLabv3, UNetBlocked


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
                            p=0.5,
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
            return DeepLabv3(
                in_channels=self.config.model.in_channels,
                out_channels=self.config.model.out_channels,
            )
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

    def create_models_name(self):
        return f"{self.config.model.name}_{self.config.model.unet_block}_{self.config.training.optimizer}_{self.config.training.lr}_{self.config.data.resize_dims}_{self.config.training.epochs}"

    def plot_predictions(
        images: Tensor,
        masks_true: Tensor,
        y_pred: Tensor,
        n_images: int = 4,
        title: str = "predictions_plot",
        segm_threshold: float = 0.5,
    ) -> None:
        y_hat = F.softmax(y_pred, dim=1)
        images = images.cpu()
        masks_true = masks_true.cpu()
        y_hat = y_hat.cpu()
        y_hat_ = torch.where(y_hat > segm_threshold, 1, 0)

        # Define the number of rows and columns
        num_rows = 3
        num_cols = n_images

        # Create a grid of subplots using GridSpec
        fig = plt.figure()
        grid = GridSpec(num_rows, num_cols + 1, figure=fig)

        # Create axes for each subplot
        axes = []
        titles = ["Image", "Mask", "Prediction\n@0.5 threshold"]
        for i in range(num_rows):
            # Add title on the left side of the row
            ax_title = fig.add_subplot(grid[i, 0])
            ax_title.set_axis_off()
            ax_title.text(0, 0.5, f"{titles[i]}", va="center")

            # Add the main subplot for each column
            row_axes = []
            data = images.permute((0, 2, 3, 1))
            cmap = None
            if i == 1:
                data = masks_true
                cmap = "gray"
            if i == 2:
                data = y_hat_
                cmap = "gray"

            for j in range(num_cols):
                ax = fig.add_subplot(grid[i, j + 1])
                ax.imshow(data[j], cmap=cmap)
                ax.set_axis_off()
                row_axes.append(ax)
            axes.append(row_axes)

        # Adjust the layout and spacing of subplots
        fig.tight_layout()

        plt.savefig(f"{title}.png")
        plt.show()
