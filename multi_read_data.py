import numpy as np
import torch
import torch.utils.data
import random
from PIL import Image
from glob import glob
import torchvision.transforms as transforms
import os

batch_w = 256
batch_h = 256


class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir, task):
        self.low_img_dir = img_dir
        self.task = task
        self.train_low_data_names = []

        for root, dirs, names in os.walk(self.low_img_dir):
            for name in names:
                self.train_low_data_names.append(os.path.join(root, name))

        self.train_low_data_names.sort()
        self.count = len(self.train_low_data_names)

        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def load_images_transform(self, file):
        im = Image.open(file).convert('RGB')
        img_norm = self.transform(im).numpy()
        img_norm = np.transpose(img_norm, (1, 2, 0))
        return img_norm

    def __getitem__(self, index):

        low = self.load_images_transform(self.train_low_data_names[index])
        if self.task == 'train':
            train_high_data_names = self.train_low_data_names[index].replace('sny', 'clear')
            high = self.load_images_transform(train_high_data_names)

        h = low.shape[0]
        w = low.shape[1]
        #
        h_offset = random.randint(0, max(0, h - batch_h - 1))
        w_offset = random.randint(0, max(0, w - batch_w - 1))

        low = np.asarray(low, dtype=np.float32)
        low = np.transpose(low[:, :, :], (2, 0, 1))
        
        if self.task == 'train':
            high = np.asarray(high, dtype=np.float32)
            high = np.transpose(high[:, :, :], (2, 0, 1))

        img_name = self.train_low_data_names[index].split('\\')[-1]
        if self.task == 'train':
            return torch.from_numpy(low), torch.from_numpy(high), img_name

        return torch.from_numpy(low), img_name

    def __len__(self):
        return self.count
