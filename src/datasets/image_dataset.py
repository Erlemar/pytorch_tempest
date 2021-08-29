from typing import List, Dict, Optional

import cv2
import numpy as np
import numpy.typing as npt
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
        Image classification dataset.

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
        # TODO rename labels and targets into one name
        if labels is not None:
            if not labels_to_ohe:
                self.labels = np.array(labels)
            else:
                self.labels = np.zeros((len(labels), n_classes))
                self.labels[np.arange(len(labels)), np.array(labels)] = 1

    def __getitem__(self, idx: int) -> Dict[str, npt.ArrayLike]:
        image_path = self.img_path + self.image_names[idx]
        image = cv2.imread(f'{image_path}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is None:
            raise FileNotFoundError(image_path)
        target = self.labels[idx]

        img = self.transforms(image=image)['image']
        sample = {'image': img, 'target': np.array(target).astype('int64')}

        return sample

    def __len__(self) -> int:
        return len(self.image_names)
