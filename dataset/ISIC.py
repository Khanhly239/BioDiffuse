from PIL import Image
from glob import glob
import numpy as np

import torch
from torch.utils.data import Dataset
import albumentations as A


class ISIC2017(Dataset):
    def __init__(self,img_dir, size):
        self.img_dir = img_dir
        self.resize = A.compose(
            [
                A.resize(size, size),
                A.ToFloat(max_value = 255),
            ]
        )
        main_data_dir = "/media/mountHDD3/data_storage/z2h/ISIC/ISIC2017" + "/ISIC-2017_Training_Data"
        self.img_paths = glob(main_data_dir + "/*.jpg")
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        image = np.array(Image.open(self._img_dir[idx]).convert("RGB"))
        resized = self.resize(image = image) 
        return torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32)