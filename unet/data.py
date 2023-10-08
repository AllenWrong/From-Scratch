from torch.utils.data import Dataset
from typing import Any
import pandas as pd
import os
from PIL import Image
import numpy as np


class SegData(Dataset):
    def __init__(self, data_root, train=True, transform = None) -> None:
        super().__init__()
        self.transform = transform
        self.img_path = os.path.join(data_root, 'train')
        self.mask_path = os.path.join(data_root, 'train_masks')
        if train:        
            self.df = pd.read_csv(os.path.join(data_root, 'train_desc.csv'))
        else:
            self.df = pd.read_csv(os.path.join(data_root, 'test_desc.csv'))

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index) -> Any:
        image_name, mask_name = self.df.iloc[index, :]

        image = Image.open(os.path.join(self.img_path, image_name)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_path, mask_name)).convert("L")
        image = np.array(image)
        mask = np.array(mask)
        mask[mask == 255.0] = 1.0
        if self.transform is not None:
            res = self.transform(image=image, mask=mask)
            image = res['image']
            mask = res['mask']

        return image, mask