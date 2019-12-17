import numpy as np
import os
from imageio import imread
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import RandomCrop, Resize, ToTensor, Normalize,RandomAffine, RandomGrayscale, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip
import random


class SSDataset_STL10(Dataset):

    def __init__(self, root_dir, in_size=64):
        self.root_dir = root_dir
        self.in_size = in_size
        self.images = []
        for r, d, f in os.walk(root_dir):
            for file in f:
                self.images.append(os.path.join(root_dir, file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = imread(self.images[idx])
        except:
            print('Unable to open image: {}'.format(idx))
        image = Image.fromarray(image)

        return image, int(0)


class ClassDataset_STL10(Dataset):

    def __init__(self, root_dir, fold_n=None):
        to_text = '../../../data/cvg/dusan/stl10/'

        folds = []
        with open(to_text+'fold_indices.txt') as f:
            for line in f.readlines():
                fold = line.split()
                folds.append(fold)
        self.root_dir = root_dir
        self.in_size = 64
        self.num_classes = 10
        self.images = []
        for r, d, f in os.walk(root_dir):
            for dir in d:
                curr_dir = os.path.join(root_dir, dir)
                for r2, d2, f2 in os.walk(curr_dir):
                    for file in f2:
                        if not fold_n==None:
                            #print(file)
                            if file.strip('.png') in folds[fold_n]:
                                self.images.append(os.path.join(curr_dir, file))
                        else:
                            self.images.append(os.path.join(curr_dir, file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = imread(self.images[idx])
        except:
            print(self.images[idx])
        transf = transforms.Compose([
            Resize(self.in_size),
            ToTensor()
        ])
        image = Image.fromarray(image)
        img = transf(image)
        label = os.path.basename(os.path.dirname(self.images[idx]))
        label = int(label) - 1
        return img, label


class RotationDataset_STL10(Dataset):

    def __init__(self, root_dir, in_size=64):
        self.root_dir = root_dir
        self.in_size = in_size
        self.images = []
        self.transf = transforms.Resize(64)
        for r, d, f in os.walk(root_dir):
            for file in f:
                self.images.append(os.path.join(root_dir, file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = imread(self.images[idx])
        except:
            print('Unable to open image: {}'.format(idx))
        image = Image.fromarray(image)
        #image = self.transf(image)
        return image, int(0)
