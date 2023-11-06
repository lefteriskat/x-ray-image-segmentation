from hydra import compose, initialize
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.data.x_ray_dataset import XRayDatasetModule


def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title("Original image", fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title("Original mask", fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title("Transformed image", fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title("Transformed mask", fontsize=fontsize)

        plt.show()


def test_dataset():
    with initialize(version_base=None, config_path="../config"):
        # config is relative to a module
        config = compose(config_name="config.yaml")

    data_module = XRayDatasetModule(config)

    train_loader, val_loader, test_loader = data_module.getDataLoaders()

    total_length = (
        len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
    )
    assert total_length == 500

def test_dataset_with_transforms():
    with initialize(version_base=None, config_path="../config"):
        # config is relative to a module
        config = compose(config_name="config.yaml")

    train_transforms = A.Compose(
                [
                    A.ToFloat(max_value=65535.0),
                    A.OneOf(
                        [
                            A.Resize(width=config.data.resize_dims, height=config.data.resize_dims, p=0.5),
                            A.RandomSizedCrop(
                                min_max_height=(50, 101), height=config.data.resize_dims, width=config.data.resize_dims, p=0.25
                            ),
                            A.PadIfNeeded(min_height=config.data.resize_dims, min_width=config.data.resize_dims, p=0.25),
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
                    A.RandomBrightnessContrast(p=0.8),
                    A.RandomGamma(p=0.8),
                    ToTensorV2(),
                ]
            )
    
    val_test_transforms = A.Compose(
        [
            A.Resize(width=config.data.resize_dims, height=config.data.resize_dims),
            ToTensorV2(),
        ]
    )


    data_module = XRayDatasetModule(config, train_transforms=train_transforms, val_transforms=val_test_transforms, test_transforms=val_test_transforms)
    train_loader, val_loader, test_loader = data_module.getDataLoaders()

    total_length = (
        len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
    )
    assert total_length == 500
