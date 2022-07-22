# -*- coding: utf-8 -*-
"""fileio

functions for manipulating image files for deep learning
"""

# Created on Tue June  09  2020
#
# Contributions and Changelog accessed via a private git repository:
# https://github.com/philipwijesinghe/deep-learning-lsm.git
# philip.wijesinghe@gmail.com
#
# Copyright (c) 2020 Philip Wijesinghe@University of St Andrews (pw64@st-andrews.ac.uk)

import glob
import numpy as np
from skimage import io
from PIL import Image


# =============================================================================
# FUNCTIONS
# =============================================================================
def save_image(datadir, date, image_no, imageLR_xz, imageHR_xz=None, mode='train'):
    """Helper function for saving images for deep learning

    Note: many modes here have been superseded and are not used

    Parameters
    ----------
    datadir
    date
    image_no
    imageLR_xz
    imageHR_xz
    mode

    Returns
    -------

    """
    img_size_xz = imageLR_xz.shape
    # Note: PIL Image operates in 'zx' coordinates
    if mode == 'paired':
        img = Image.new('L', (img_size_xz[0] * 2, img_size_xz[1]))
        img.paste(Image.fromarray(imageLR_xz.transpose()), (0, 0))
        img.paste(Image.fromarray(imageHR_xz.transpose()), (img_size_xz[0], 0))
        img.save(datadir + '\\paired_data\\' + '%05i.png' % image_no)
    elif mode == 'train' or mode == 'val' or mode == 'test':
        img = Image.new('L', (img_size_xz[0] * 2, img_size_xz[1]))
        img.paste(Image.fromarray(imageLR_xz.transpose()), (0, 0))
        img.paste(Image.fromarray(imageHR_xz.transpose()), (img_size_xz[0], 0))
        img.save(datadir + '\\' + date + '\\' + mode + '\\' + '%05i.png' % image_no)
    elif mode == 'process':
        img = Image.new('L', (img_size_xz[0], img_size_xz[1]))
        img.paste(Image.fromarray(imageLR_xz.transpose()), (0, 0))
        img.save(datadir + '\\' + date + '\\process\\' + '%05i.png' % image_no)
        if imageHR_xz is not None:
            img = Image.new('L', (img_size_xz[0], img_size_xz[1]))
            img.paste(Image.fromarray(imageHR_xz.transpose()), (0, 0))
            img.save(datadir + '\\' + date + '\\reference\\' + '%05i.png' % image_no)
    elif mode == 'reference':
        img = Image.new('L', (img_size_xz[0], img_size_xz[1]))
        img.paste(Image.fromarray(imageLR_xz.transpose()), (0, 0))
        img.save(datadir + '\\' + date + '\\reference\\' + '%05i.png' % image_no)
    else:
        print("Incorrect mode")

    return


def load_img_stack(folder, image_nos=0):
    """Loads image sequence in folder as a numpy image stack

    Parameters
    ----------
    folder
        directory containing png or tiff images
    image_nos
        if 0, loads all images; if a list, loads specified images (by index)

    Returns
    -------
    numpy array of size [i, x, y], where i is image index

    """
    print(folder)
    if folder[-3:] == 'tif':
        img_stack = io.imread(folder)
        return img_stack

    files = glob.glob(folder + '/*.png')
    files += glob.glob(folder + '/*.tif')

    if image_nos == 0:
        image_nos = np.arange(len(files))

    img = io.imread(files[0])
    img_size = img.shape
    img_stack = np.zeros([len(image_nos), img_size[0], img_size[1]]).astype(img.dtype)

    for i, v in enumerate(image_nos):
        img_stack[i, :, :] = io.imread(files[v])
        
    return img_stack
