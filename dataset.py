import torch
from torch.utils.data.dataset import Dataset
import os
import cv2
import numpy as np

from utils import normalize_image

class NetDataset(Dataset):
    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self.input_shape = (512, 288)
        self.data = self.read_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = np.array(self.data[item]).astype(np.float32)
        return torch.from_numpy(data)

    @property
    def read_data(self):
        print(f"Dataset Path: {self.folder}")
        files = os.listdir(self.folder)
        # print("Reading data... ", end='')

        data = []
        total = 0
        for file in files:
            # Add image and preprocess
            image = cv2.imread(os.path.join(self.folder, file))
            image = cv2.resize(image, self.input_shape)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.transpose(normalize_image(image), (2, 0, 1))

            data.append(image)
            total += 1
            print(f"\rReading data... [{total} / {len(files)}] ", end='')

        print("Completed! ")
        return data