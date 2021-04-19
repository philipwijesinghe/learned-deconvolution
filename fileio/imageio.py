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

from PIL import Image


# =============================================================================
# FUNCTIONS
# =============================================================================
def save_image(datadir, date, image_no, imageLR_xz, imageHR_xz=None, mode='train'):
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
