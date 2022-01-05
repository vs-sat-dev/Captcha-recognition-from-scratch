import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import numpy as np


class CaptchaDataset(Dataset):
    def __init__(self, data, chars_dict, transforms=None):
        self.images = data
        self.chars_dict = chars_dict
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(f'data/{self.images[index]}')
        if self.transforms:
            img = self.transforms(img)
        else:
            img = torch.tensor(np.array(img))

        img = img / 255.0

        label_temp = self.images[index].split('.')[0]
        label = [self.chars_dict[char] for char in label_temp]

        return img.permute(2, 0, 1)[:3, :, :], torch.tensor(label, dtype=torch.long)


def get_data(path):
    data = os.listdir(path)
    data_train, data_test = train_test_split(data, test_size=0.1)
    chars_set = set()
    for chars in data:
        chars_splited = chars.split('.')[0]
        for char in chars_splited:
            chars_set.add(char)
    return data_train, data_test, chars_set
