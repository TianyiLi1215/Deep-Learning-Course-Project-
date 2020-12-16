import numpy as np
import random
import glob
import os
import time
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=True, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_X = sorted(glob.glob(os.path.join(root, '%s/X' % mode) + '/*.*'))
        self.files_Y = sorted(glob.glob(os.path.join(root, '%s/Y' % mode) + '/*.*'))

    def __getitem__(self, index):
        image_X = self.transform(Image.open(self.files_X[index % len(self.files_X)]))

        if self.unaligned:
            image_Y = self.transform(Image.open(self.files_Y[random.randint(0, len(self.files_Y) - 1)]))
        else:
            image_Y = self.transform(Image.open(self.files_Y[index % len(self.files_Y)]))

        return {"X": image_X, "Y": image_Y}

    def __len__(self):
        return max(len(self.files_X), len(self.files_Y))


class ReplayBuffer(object):
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.size = 0

    def add_sample(self, img, batch_size):
        if random.random() > 0.5 or self.size < 1:
            ret = img
        else:
            ret = random.sample(self.buffer, batch_size)
            ret = torch.cat(ret, 0)

        if self.size < self.capacity:
            self.buffer.append(img)
        else:
            self.buffer[self.position] = img
        self.position = (self.position + 1) % self.capacity
        self.size += 1

        return ret


class LambdaLR():
    def __init__(self, n_epochs, offset, decay):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay = decay

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay)/(self.n_epochs - self.decay)







