import glob
import logging
import os

import numpy as np
import PIL.Image as Image
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, Dataset


class XRayDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.transforms = transforms
        self.data_path = data_dir
        # use glob to take the labels and the data
        self.data_paths = sorted(
            glob.glob(os.path.join(self.data_path + "/data/*.tiff"))
        )
        self.label_paths = sorted(
            glob.glob(os.path.join(self.data_path + "/labels/*.tif"))
        )

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(data_path).resize((512, 512))
        mask = Image.open(label_path).resize((512, 512))

        image_final = np.array(image)
        mask_final = np.array(mask)

        if self.transforms:
            augmented = self.transforms(image=image_final, mask=mask_final)
            image_final = augmented["image"]
            mask_final = augmented["mask"]

        # make the class values from 0 128 255 to 0 1 2
        # will be needed for one-hot encoding
        # mask_aug[mask_aug == 128] = 1.0
        # mask_aug[mask_aug == 255] = 2.0

        return image_final, mask_final

    # def __getitemtest__(self, idx):
    #     'Generates one sample of data'
    #     data_path = self.data_paths[idx]
    #     label_path = self.label_paths[idx]

    #     image = Image.open(data_path).resize((512,512))
    #     mask = Image.open(label_path).resize((512,512))

    #     image_np = np.array(image)
    #     mask_np = np.array(mask)

    #     image_final = image_np
    #     mask_final = mask_np

    #     # image = cv2.imread(data_path, cv2.IMREAD_UNCHANGED)
    #     # mask = cv2.imread(data_path, cv2.IMREAD_UNCHANGED)

    #     print(f"image_shape:{image.size}, mask_shape: {mask.size}")
    #     if self.augment:
    #         augmented = self.augmentations(image=image_np, mask=mask_np)
    #         image_final = augmented['image']
    #         mask_final = augmented['mask']

    #     return image_final,mask_final#,X, Y

class XRayDatasetModule:
    def __init__(self, config: DictConfig, train_transforms=None, val_transforms=None, test_transforms=None):
        self.config = config
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms 
        self.test_transforms = test_transforms  

    def createDataLoaders(
        self,
        data_dir: str,
        batch_size=32,
        train_fraction: float = 0.8,
        val_fraction: float = 0.15,
        test_fraction: float = 0.05,
    ):
        logger = logging.getLogger(__name__)
        logger.info("Creating data loaders")

        train_dataset: torch.utils.data.Dataset = None
        val_dataset: torch.utils.data.Dataset = None
        test_dataset: torch.utils.data.Dataset = None

        train_dataset = XRayDataset(data_dir=data_dir, transforms=self.train_transforms)
        val_dataset = XRayDataset(data_dir=data_dir, transforms=self.val_transforms)
        test_dataset = XRayDataset(data_dir=data_dir, transforms=self.test_transforms)

        indices = torch.randperm(len(train_dataset))
        val_size = int(np.floor(len(train_dataset) * val_fraction))
        test_size = int(np.floor(len(train_dataset) * test_fraction))

        train_dataset = torch.utils.data.Subset(
            train_dataset, indices[: -(val_size + test_size)]
        )
        val_dataset = torch.utils.data.Subset(val_dataset, indices[-val_size:])
        test_dataset = torch.utils.data.Subset(
            test_dataset, indices[-(val_size + test_size) : -val_size]
        )

        logger.info(f"Train length: {len(train_dataset)}")
        logger.info(f"Validation length: {len(val_dataset)}")
        logger.info(f"Test length: {len(test_dataset)}")

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.config.data.num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True, num_workers=self.config.data.num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, num_workers=self.config.data.num_workers
        )

        return train_loader, val_loader, test_loader
    
    def getDataLoaders(self):
        return self.createDataLoaders(data_dir=self.config.data.data_path, batch_size=self.config.data.batch_size,
                                      train_fraction=self.config.data.train_size,
                                      val_fraction=self.config.data.val_size,
                                      test_fraction=self.config.data.test_size,
                                      )
