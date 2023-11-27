import albumentations as A
import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from matplotlib.gridspec import GridSpec
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
import os
from src.models.unet import DeepLabv3, UNetBlocked, deeplabv3_smp, deeplabv3plus_smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, TverskyLoss


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
                A.GaussNoise(p=0.5),
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
        elif self.config.model.name == "deeplabsmp":
            return deeplabv3_smp()
        elif self.config.model.name == "deeplabsmp+":
            return deeplabv3plus_smp()
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
        if self.config.model.loss == "ce+":
            return nn.CrossEntropyLoss()         #DiceLoss(‘multiclass’), FocalLoss(‘multiclass’), TverskyLoss(‘multiclass’)
        elif self.config.model.loss == "dice":
            return DiceLoss('multiclass')
        elif self.config.model.loss == "focal":
            return FocalLoss('multiclass')
        else: 
            return TverskyLoss('multiclass')

    def create_models_name(self):
        return f"{self.config.model.name}_{self.config.model.unet_block}_{self.config.training.optimizer}_{self.config.training.lr}_{self.config.data.resize_dims}_{self.config.training.epochs}"

    def transform_prediction(self, pred):
        # This function is used to transform the prediction tensor from (1,3, width, height) to (1, width, height). 
        # Also it refactors the pixels that were alterned in the x_ray_dataset.py
        pred_softmax = torch.nn.functional.softmax(pred, dim=1)
        pred_mask = torch.argmax(pred_softmax, dim=1)
        pred_mask[pred_mask == 1] = 128
        pred_mask[pred_mask == 2] = 255
        return pred_mask.long()

    def plot_predictions(self, data_batch, predictions, label_batch = None, counter=0, model_name="dummy"):     
        # Convert tensors to NumPy arrays
        data_np = data_batch.cpu().numpy()
        predictions = self.transform_prediction(predictions)
        pred_np = predictions.cpu().numpy()

        print(f"Data : {data_batch.size()}")
        print(f"Pred : {predictions.size()}")

        if label_batch is not None:
            label_np = label_batch.cpu().numpy()
            print(f"Label : {label_batch.size()}")

        batches_len = data_np.shape[0]
        
        number_of_figures = 2 if label_batch is None else 3 
        
        fig, axs = plt.subplots(number_of_figures, batches_len, figsize=(8, 10))
        axs = axs.flatten()
        # Original Data Image
        for i in range(batches_len):
            axs[i].imshow(data_np[i, 0], cmap="gray")
            axs[i].axis("off")
            axs[i].set_title("Data")
            
        # Predicted Label Images
        for i in range(batches_len):
            axs[i + batches_len].imshow(pred_np[i, :, :], cmap="gray")
            axs[i + batches_len].axis("off")
            axs[i + batches_len].set_title("Predicted")

        # Original Label Image
        if label_batch is not None:
            for i in range(batches_len):
                axs[i + 1 + batches_len].imshow(label_np[i, :, :], cmap="gray")
                axs[i + 1 + batches_len].axis("off")
                axs[i + 1 + batches_len].set_title("Label")

       
        dir_path = os.path.join("reports/figures/", model_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.savefig( dir_path + f"/{model_name}_predictions_{counter}.png")
        
    
    def plot_probability_mask(self, data_batch, predictions, label_batch = None, counter=0, model_name="dummy"):     
        # Convert tensors to NumPy arrays
        data_np = data_batch.cpu().numpy()
        predictions = F.softmax(predictions, dim=1)
        predictions = torch.max(predictions, dim=1).values
        pred_np = predictions.cpu().numpy()

        print(f"Data : {data_batch.size()}")
        print(f"Pred : {predictions.size()}")

        if label_batch is not None:
            label_np = label_batch.cpu().numpy()
            print(f"Label : {label_batch.size()}")

        batches_len = data_np.shape[0]
        
        number_of_figures = 2 if label_batch is None else 3 
        
        fig, axs = plt.subplots(number_of_figures, batches_len, figsize=(8, 10))
        axs = axs.flatten()
        # Original Data Image
        for i in range(batches_len):
            axs[i].imshow(data_np[i, 0], cmap="gray")
            axs[i].axis("off")
            axs[i].set_title("Data")
            
        # Predicted Label Images
        for i in range(batches_len):
            axs[i + batches_len].imshow(pred_np[i, :, :], cmap="Blues")
            axs[i + batches_len].axis("off")
            axs[i + batches_len].set_title("Predicted")

        # Original Label Image
        if label_batch is not None:
            for i in range(batches_len):
                axs[i + 1 + batches_len].imshow(label_np[i, :, :], cmap="gray")
                axs[i + 1 + batches_len].axis("off")
                axs[i + 1 + batches_len].set_title("Label")

       
        dir_path = os.path.join("reports/figures/", model_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.savefig( dir_path + f"/{model_name}_probabilities_{counter}.png")