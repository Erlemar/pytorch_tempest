from typing import List, Dict, Optional

import cv2
import numpy as np
from albumentations.core.composition import Compose
from torch.utils.data import Dataset


class ImageClassificationDataset(Dataset):
    def __init__(
        self,
        image_names: List,
        transforms: Compose,
        labels: Optional[List[int]],
        img_path: str = '',
        mode: str = 'train',
        labels_to_ohe: bool = False,
        n_classes: int = 1,
    ):
        """
        Prepare data for wheat competition.

        Args:
            df: dataframe with image id and bboxes
            mode: train/val/test
            img_path: path to images
            transforms: albumentations
        """

        self.mode = mode
        self.transforms = transforms
        self.img_path = img_path
        self.image_names = image_names
        # print('labels_to_ohe', labels_to_ohe)
        if labels:
            if not labels_to_ohe:
                self.labels = np.array(labels)
            else:
                self.labels = np.zeros((len(labels), n_classes))
                self.labels[np.arange(len(labels)), np.array(labels)] = 1

    def __getitem__(self, idx: int) -> Dict[str, np.array]:
        image_path = self.img_path + self.image_names[idx]
        image = cv2.imread(f'{image_path}', cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        target = self.labels[idx]

        img = self.transforms(image=image)['image']
        sample = {'image': img, 'target': np.array(target).astype('int64')}
        # print(sample)
        # print(img.shape, target.shape, image_path)

        return sample

    def __len__(self) -> int:
        return len(self.image_names)
