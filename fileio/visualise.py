# -*- coding: utf-8 -*-
"""visualise

functions for loading, slicing and saving network images
"""

# Created on 20210201
#
# Contributions and Changelog accessed via a private git repository:
# https://github.com/philipwijesinghe/deep-learning-lsm.git
# philip.wijesinghe@gmail.com
#
# Copyright (c) 2021 Philip Wijesinghe@University of St Andrews (pw64@st-andrews.ac.uk)


import os
import glob
import sys
sys.path.append('../')

from PIL import Image
import numpy as np
import matplotlib.pyplot as mpl

# from fileio.imageio import


class Opts():
    def __init__(self):
        # Options class, adressable as Opts.property
        self.pix_size_zxy = [0.85, 0.85, 0.35357416468712344]
        # self.pix_size_zxy = [0.85, 0.85, 0.1967724576274877]
        self.spacing_pix_yz = [250, 25]


# =============================================================================
# FUNCTIONS
# =============================================================================
def visualise_net_in_out(parentdir, opts):
    """
    visualise_net_in_out(parentdir)

    Parameters
    ----------
    parentdir : String
        Directory housing Process- and Processed- image slices

    Returns
    -------
    None.

    """

    # Find process (IN) folder
    dirs_process = find_folder_dirs(parentdir, 'process-')
    dirs_precessed = find_folder_dirs(parentdir, 'processe')

    for dir_ in dirs_process:
        if dir_:
            load_save_stack(dir_, opts)
    for dir_ in dirs_precessed:
        if dir_:
            load_save_stack(dir_, opts)


def find_folder_dirs(parentdir, pattern):
    """
    dirs = find_folder_dirs(parentdir, pattern)

    find a list of directories in the parentdir that match the pattern:
        <pattern>-*

    Parameters
    ----------
    parentdir : Paterent directory string
    pattern : String, pattern, eg 'process'

    Returns
    -------
    dirs : list of strings, absolute path of directories

    """

    dirs = []
    for file in glob.glob(os.path.join(parentdir, pattern + '*[!-out]')):
        dirs.append(file)

    return dirs


def load_save_stack(folderdir, opts):
    print('Slicing %s' % folderdir)

    outdir = folderdir + '-out'
    os.makedirs(os.path.join(outdir, 'zx'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'xy'), exist_ok=True)

    imgs = []
    for file in glob.glob(os.path.join(folderdir, '*.png')):
        imgs.append(file)

    n_images = len(imgs)

    def load_img(imgpath):
        return np.array(Image.open(imgpath))

    template = load_img(imgs[0])
    img_size = template.shape
    mpl.imshow(template)

    img_stack = np.zeros([img_size[0], img_size[1], n_images], dtype=template.dtype)
    for i, img_ in enumerate(imgs):
        img_stack[:, :, i] = load_img(img_)

    # Slice zx images
    idx_y = np.arange(0, n_images - 1, opts.spacing_pix_yz[0])
    for idx in idx_y:
        img_ = img_stack[:, :, idx]
        img_ = Image.fromarray(img_, 'L')
        img_.save(os.path.join(outdir, 'zx', 'zx_%s.png' % idx))

    # Slice xy images
    idx_z = np.arange(0, img_size[0] - 1, opts.spacing_pix_yz[1])
    out_size = (int(n_images * opts.pix_size_zxy[2] / opts.pix_size_zxy[1]), img_size[1])
    for idx in idx_z:
        img_ = img_stack[idx, :, :]
        img_ = Image.fromarray(img_, 'L')
        img_ = img_.resize(out_size)
        img_.save(os.path.join(outdir, 'xy', 'xy_%s.png' % idx))


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Run this section as a script, while making all defined functions
    # available to the module

    opts = Opts()

    parentdir = r'E:\LSM-deeplearning\20201008_BlastocystJC1_Processed\BlastocystJC1\150Radius-200exposure-0.5step-100mW-alpha1_1'

    visualise_net_in_out(parentdir, opts)
