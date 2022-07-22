# -*- coding: utf-8 -*-
"""datasets

Defines the dataloaders using PyTorch inherited Dataset class
"""

# Contributions and Changelog accessed via a private git repository:
# https://github.com/philipwijesinghe/deep-learning-lsm.git
# philip.wijesinghe@gmail.com

# Copyright (c) 2020 Philip Wijesinghe@University of St Andrews (pw64@st-andrews.ac.uk)

import glob
import os
import numpy as np
# [PYTORCH]
# import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
# import matplotlib.pyplot as plt
from PIL import Image


# =============================================================================
# DATASET CLASSES
# =============================================================================
class ImageDataset(Dataset):
    """Loads paired images in /<mode> subfolder."""

    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)


class ImageDatasetRef(Dataset):
    """Loads single images in /reference subfolder."""

    def __init__(self, root, transforms_=None, mode="reference"):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert('L')

        img = self.transform(img)

        return {"REF": img}

    def __len__(self):
        return len(self.files)


class ImageDatasetLSM(Dataset):
    """Loads horizontally merged paired images in /'mode' subfolder."""

    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert('L')
        w, h = img.size
        img_LR = img.crop((0, 0, w / 2, h))
        img_HR = img.crop((w / 2, 0, w, h))

        img_LR = self.transform(img_LR)
        img_HR = self.transform(img_HR)

        return {"LR": img_LR, "HR": img_HR}

    def __len__(self):
        return len(self.files)


class ImageDatasetTrainRef(Dataset):
    """
    Loads LR images from horizontally merged paired images in /'mode' subfolder.
    The HR images are taken from a /reference folder
    """

    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        self.files_HR = sorted(glob.glob(os.path.join(root, 'reference') + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert('L')
        w, h = img.size
        img_LR = img.crop((0, 0, w / 2, h))

        img_HR = Image.open(self.files_HR[index % len(self.files)]).convert('L')

        img_LR = self.transform(img_LR)
        img_HR = self.transform(img_HR)

        return {"LR": img_LR, "HR": img_HR}

    def __len__(self):
        return len(self.files)


class ImageDatasetWidefield(Dataset):
    """Loads widefield images and partitions them into small tiles.
    Enables restitching through tile_coordinates."""

    def __init__(self, root, img_size, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.files = sorted(glob.glob(os.path.join(root, 'process') + "/*.*"))

        img = Image.open(self.files[0 % len(self.files)]).convert('L')
        self.w, self.h = img.size

        # number of mosaic via ceiling
        self.nw = -(-self.w // img_size)
        self.nh = -(-self.h // img_size)
        self.tiles_per_img = self.nw * self.nh
        # image padding
        self.width_full_size = self.nw * img_size
        self.height_full_size = self.nh * img_size
        self.width_pad_to_full = self.width_full_size - self.w
        self.height_pad_to_full = self.height_full_size - self.h

    def __getitem__(self, index):
        # figure out index
        index_img = index // self.tiles_per_img
        index_tile = index % self.tiles_per_img
        cw, ch = self.tile_coordinates(index_tile)

        # Load parent image
        img = Image.open(self.files[index_img % len(self.files)]).convert('L')
        # plt.imshow(img)

        # Tensor [C, H, W]
        img = self.transform(img)

        # pad image tensor to a multiple of img_size
        img.unsqueeze_(0)
        img = F.pad(input=img,
                    pad=(0, self.width_pad_to_full, 0, self.height_pad_to_full),
                    mode='reflect')
        img.squeeze_(0)

        # crop to the tile
        img_A = img[:, ch:self.img_size + ch, cw:self.img_size + cw]

        return {"A": img_A}

    def __len__(self):
        return len(self.files) * self.tiles_per_img

    def tile_coordinates(self, index_tile):
        index_w = index_tile % self.nw
        index_h = index_tile // self.nw
        cw = self.img_size * index_w
        ch = self.img_size * index_h
        return cw, ch


class ImageDatasetWidefieldStitch(Dataset):
    """Loads widefield images and partitions them into small tiles.
    Includes restitchig method."""

    def __init__(self, root, img_size, margin, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.margin = margin
        # self.files = sorted(glob.glob(os.path.join(root, 'process') + "/*.*"))
        self.files = sorted(glob.glob(root + "/*.png"))

        img = Image.open(self.files[0 % len(self.files)]).convert('L')
        self.w, self.h = img.size

        # number of mosaic via ceiling: ceil(a/b) = -(-a//b)
        self.nw = -(-(self.w - 2 * self.margin) // (img_size - 2 * self.margin))
        self.nh = -(-(self.h - 2 * self.margin) // (img_size - 2 * self.margin))
        self.tiles_per_img = self.nw * self.nh

        # image padding
        self.width_full_size = (self.nw * img_size - 2 * (self.nw - 1) * self.margin)
        self.height_full_size = (self.nh * img_size - 2 * (self.nh - 1) * self.margin)

        self.width_pad_to_full = self.width_full_size - self.w
        self.height_pad_to_full = self.height_full_size - self.h

    def __getitem__(self, index):
        # figure out index
        index_img = index // self.tiles_per_img
        index_tile = index % self.tiles_per_img
        cw, ch = self.tile_coordinates(index_tile)

        # Load parent image
        img = Image.open(self.files[index_img % len(self.files)]).convert('L')
        # plt.imshow(img)

        # Tensor [C, H, W]
        img = self.transform(img)  # .point(lambda i: i * 1.5)

        # pad image tensor to a multiple of img_size
        img.unsqueeze_(0)
        img = F.pad(input=img,
                    pad=(0, self.width_pad_to_full, 0, self.height_pad_to_full),
                    mode='reflect')
        img.squeeze_(0)

        # crop to the tile
        img_A = img[:, ch:self.img_size + ch, cw:self.img_size + cw]

        return {"A": img_A}

    def __len__(self):
        return len(self.files) * self.tiles_per_img

    def tile_coordinates(self, index_tile):
        index_w = index_tile % self.nw
        index_h = index_tile // self.nw

        cw = index_w * (self.img_size - 2 * self.margin)
        ch = index_h * (self.img_size - 2 * self.margin)

        return cw, ch

    def tile_to_stitch(self, img, index_tile):
        index_w = index_tile % self.nw
        index_h = index_tile // self.nw

        # start index
        sw = index_w * (self.img_size - 2 * self.margin) + self.margin
        sh = index_h * (self.img_size - 2 * self.margin) + self.margin
        if index_w == 0:
            sw = 0
        if index_h == 0:
            sh = 0

        # end index
        ew = sw + self.img_size - 2 * self.margin
        eh = sh + self.img_size - 2 * self.margin
        if index_w == self.nw - 1:
            ew = ew + self.margin
        if index_h == self.nh - 1:
            eh = eh + self.margin

        (left, upper, right, lower) = (
            self.margin if index_w != 0 else 0,
            self.margin if index_h != 0 else 0,
            self.img_size - self.margin if index_w != self.nw - 1 else self.img_size,
            self.img_size - self.margin if index_h != self.nh - 1 else self.img_size
        )

        img = img.crop((left, upper, right, lower))

        return img, sw, sh, ew, eh
