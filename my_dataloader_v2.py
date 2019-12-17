from __future__ import print_function
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
# from Places205 import Places205
import numpy as np
import random
import torchnet as tnt
from PIL import Image
from torch.utils.data import _utils
default_collate = _utils.collate.default_collate


class MyDataLoader():
    def __init__(self,
                 dataset,
                 batch_size=1,
                 unsupervised=True,
                 epoch_size=None,
                 num_workers=0,
                 shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.in_size = dataset.in_size
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.num_workers = num_workers

        dim1 = random.choice(range(64, 129, 1))
        self.transf_org = transforms.Compose([
            transforms.Resize(dim1),
            transforms.RandomCrop(self.in_size),
            transforms.RandomGrayscale(0.5),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        dim2 = random.choice(range(64, 129, 1))
        while dim1 == dim2:
            dim2 = random.choice(range(64, 129, 1))
        self.transf_new = transforms.Compose([
            transforms.Resize(dim2),
            transforms.RandomCrop(self.in_size),
            transforms.RandomGrayscale(0.5),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)
        # if in unsupervised mode define a loader function that given the
        # index of an image it returns the 4 rotated copies of the image
        # plus the label of the rotation, i.e., 0 for 0 degrees rotation,
        # 1 for 90 degrees, 2 for 180 degrees, and 3 for 270 degrees.

        def _load_function(idx):
            idx = idx % len(self.dataset)
            img0, _ = self.dataset[idx]
            org_img = img0

            rotated_imgs_org = [
                self.transf_org(org_img),
                self.transf_org(org_img)
            ]
            org = torch.stack(rotated_imgs_org, dim=0)

            idx_fake = idx
            while idx == idx_fake:
                idx_fake = int(np.random.rand(1)*len(self.dataset))
            image_fake, _ = self.dataset[idx_fake]
            imgs_new = [
                self.transf_new(org_img),
                self.transf_new(image_fake)
            ]
            new = torch.stack(imgs_new, dim=0)
            labels = torch.LongTensor([1, 0])

            return (org, new), labels

        def _collate_fun(batch):
            batch = default_collate(batch)
            assert(len(batch) == 2)
            org, new = batch[0]
            batch_size, rotations, channels, height, width = org.size()
            org = org.view([batch_size*rotations, channels, height, width])
            batch_size, rotations, channels, height, width = new.size()
            new = new.view([batch_size*rotations, channels, height, width])
            batch[0] = (org, new)
            batch_size, rotations = batch[1].size()
            batch[1] = batch[1].view([batch_size*rotations])
            return batch

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
                                              load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
                                           collate_fn=_collate_fun, num_workers=self.num_workers,
                                           shuffle=self.shuffle)
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size / self.batch_size
