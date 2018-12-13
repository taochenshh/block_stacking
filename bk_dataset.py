'''
Author: Tao Chen (CMU RI)
Date: 11/25/2018
'''

import json
import os

from PIL import Image
from torch.utils.data import Dataset


class BKDataset(Dataset):
    def __init__(self, mode, transform=None):
        self.data_dir = '../data'
        with open('data_split.json', 'r') as f:
            split_idx = json.load(f)
        self.img_folders = [os.path.join(self.data_dir, folder)
                            for folder in split_idx[mode]]
        self.transform = transform

    def __len__(self):
        return len(self.img_folders)

    def __getitem__(self, idx):
        img_file = os.path.join(self.img_folders[idx], 'img.png')
        label_file = os.path.join(self.img_folders[idx], 'label.json')
        image = Image.open(img_file)
        with open(label_file, 'r') as f:
            label = int(json.load(f))
        if self.transform is not None:
            image = self.transform(image)
        return image, label
