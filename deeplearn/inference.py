# -*- coding: utf-8 -*-
""" Processes data with a trained model
"""

# Created on 20210128

# Contributions and Changelog accessed via a private git repository:
# https://github.com/philipwijesinghe/deep-learning-lsm.git
# philip.wijesinghe@gmail.com

# Copyright (c) 2021 Philip Wijesinghe@University of St Andrews (pw64@st-andrews.ac.uk)

import os
import numpy as np
import sys
sys.path.append('../')

import torch

import torchvision.transforms as transforms
import deeplearn.physics_based_training as dl

from torch.utils.data import DataLoader
# from torchvision import datasets
from torch.autograd import Variable

from deeplearn.models import GeneratorUNet256_4x, GeneratorUNet64US, GeneratorUNet128_x2
from deeplearn.models import GeneratorResNet
from deeplearn.datasets import Image, ImageDatasetWidefieldStitch


# =============================================================================
# Process a stack of widefield images with a trained network
# =============================================================================
def process_widefield_stack(in_dir, model_dir, conf):
    """
    Process a stack of widefield images with a trained network

    Parameters
    ----------
    in_dir :    directory to stack of images (.png)

    model_dir :     path to saved model

    config :    config structure returned from model (physics-based-training)

    Returns
    -------

    """

    # =============================================================================
    # Manual parameters (for now)
    # =============================================================================
    margin = 16
    bsize = 8  # batch size for processing

    # =============================================================================
    # Main
    # =============================================================================
    parent_dir = os.path.split(in_dir)[0]
    save_dir = os.path.join(parent_dir, "Processed")
    os.makedirs(save_dir, exist_ok=True)

    generator_dir = os.path.join(
        model_dir,
        'saved_models',
        'generator_%d.pth' % (conf.n_epochs - 1)
    )

    # Load DL model
    cuda = True if torch.cuda.is_available() else False
    # cuda = False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Initialize generator and discriminator
    if conf.img_size == 64:
        if conf.model.lower() == 'unet':
            generator = GeneratorUNet64US(in_channels=conf.channels,
                                          out_channels=conf.channels)
        elif conf.model.lower() == 'resnet':
            generator = GeneratorResNet(in_channels=conf.channels,
                                        out_channels=conf.channels)
        else:
            print("No compatible model specified")
            return
    elif conf.img_size == 128:
        generator = GeneratorUNet128_x2(in_channels=conf.channels,
                                        out_channels=conf.channels)
    elif conf.img_size == 256:
        generator = GeneratorUNet256_4x(in_channels=conf.channels,
                                        out_channels=conf.channels)
    else:
        print("No model for image size specified")
        return

    if cuda:
        generator = generator.cuda()

    # Load state dictionary
    generator.load_state_dict(torch.load(generator_dir))
    generator.eval()

    # Data loader and transforms
    transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
    dataset = ImageDatasetWidefieldStitch(
        root=in_dir,
        img_size=conf.img_size,
        margin=margin,
        transforms_=transforms_
    )
    dataloader = DataLoader(
        dataset,
        batch_size=dataset.tiles_per_img,  # Batch process each tile in an image
    )

    b_per_wide = -(-dataset.tiles_per_img // bsize)  # Ceiling

    # Process images through the model
    for i, batch in enumerate(dataloader):
        # t0 = timer()
        # Batch process each tile in an image
        real_A = Variable(batch["A"].type(Tensor)).detach()
        fake_B = real_A

        # Process each tile separately (minimal GPU memory usage)
        # for bb in range(dataset.tiles_per_img):
        #     fake_B[bb:bb+1, :, :, :] = generator(real_A[bb:bb+1, :, :, :]).detach()

        for bb in range(b_per_wide):
            b1 = bb * bsize
            b2 = (bb + 1) * bsize
            b2 = np.min([b2, dataset.tiles_per_img])
            fake_B[b1:b2, :, :, :] = generator(real_A[b1:b2, :, :, :]).detach()

        # Assemble mosaic
        img_wide = Image.new('L', (dataset.width_full_size, dataset.height_full_size))
        for ti in range(dataset.tiles_per_img):
            img_out = fake_B.data[ti, :, :, :]

            # To UINT8 image;
            # model output is passed through a tanh activation
            # thus, output tensor is in [-1, 1]
            img_out = img_out.cpu().squeeze_(0).numpy()
            img_out = 255 * (img_out + 1) / 2
            img_pil = Image.fromarray(img_out.astype('uint8'), mode='L')

            # cw, ch = dataset.tile_coordinates(ti)
            img_paste, sw, sh, ew, eh = dataset.tile_to_stitch(img_pil, ti)
            img_wide.paste(img_paste, (sw, sh))

        # crop to original size
        img_wide = img_wide.crop((0, 0, dataset.w, dataset.h))
        img_wide.save(save_dir + "/%04i.png" % i)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Run this section as a script, while making all defined functions
    # available to the module

    config = dl.Config()

    in_dir = r'E:\LSM-deeplearning\20201009_BlastocystNS_Processed\BlastocystNS\150Radius-200exposure-0.5step-100mW-alpha1_1\Process-20201015'
    model_dir = r'E:\LSM-deeplearning\TrainedModels\GoodModels\20210113_Airy_g1_Blastocyst_l30_b1e-4'

    process_widefield_stack(in_dir, model_dir, config)
