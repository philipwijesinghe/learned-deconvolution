# -*- coding: utf-8 -*-
"""Prepare data for training based on a specified config file
"""

# Created on Wed June 17 2020

# Contributions and Changelog accessed via a private git repository:
# https://github.com/philipwijesinghe/deep-learning-lsm.git
# philip.wijesinghe@gmail.com

# Copyright (c) 2020 Philip Wijesinghe@University of St Andrews (pw64@st-andrews.ac.uk)

import os
import shutil
import sys
sys.path.append('../')
import glob
import numpy as np

from fileio.yamlio import load_config
from lsm.simulate import simulate_lsm_vectorial, simulate_lsm_convolution

# for repeatability, set RNG seed
# np.random.seed(10)


# =============================================================================
# FUNCTIONS
# =============================================================================
def prepare_training_data(datadir, overwrite, **kwargs):
    train_real = True
    if 'data' in kwargs:
        if kwargs['data'] == 'sim':
            train_real = False

    config, psfconfigLR, psfconfigHR = load_config(datadir + '/TrainConfig.yml')
    rootdir = config['dir']
    phys_dirs = config['phys_dirs'].split()
    if train_real:
        real_dirs = config['real_dirs'].split()

    # Prepare folders
    if train_real:
        dir_real = os.path.join(rootdir, 'real')
    dir_train = os.path.join(rootdir, 'train')
    dir_val = os.path.join(rootdir, 'val')

    if os.path.exists(dir_train):
        if overwrite:
            if train_real:
                shutil.rmtree(dir_real)
            shutil.rmtree(dir_train)
            shutil.rmtree(dir_val)
        else:
            return

    if train_real:
        os.makedirs(dir_real, exist_ok=False)
    os.makedirs(dir_train, exist_ok=False)
    os.makedirs(dir_val, exist_ok=False)

    # Count total phys images
    n_images_phys = 0
    for dir_ in phys_dirs:
        file_list = [f for f in glob.glob(dir_ + "\\*.png")]
        n_images_phys += len(file_list)

    # Validation shuffler
    n_val = config['n_val']
    rperm = np.random.permutation(n_images_phys)
    rperm = rperm[:n_val]

    # Copy phys images
    n_offset = 0
    for dir_ in phys_dirs:
        (_, _, files) = next(os.walk(dir_), (None, None, []))
        files = [f for f in files if f.endswith(".png")]

        n_images = len(files)
        img_shuffle = np.random.permutation(n_images)

        for i, v in enumerate(files):
            img_name = ('%05i.png' % img_shuffle[i])
            if n_offset + i in rperm:
                shutil.copyfile(os.path.join(dir_, v),
                                os.path.join(dir_val, img_name))
            else:
                shutil.copyfile(os.path.join(dir_, v),
                                os.path.join(dir_train, img_name))

        n_offset += n_images

    # Copy real images
    if train_real:
        for dir_ in real_dirs:
            (_, _, files) = next(os.walk(dir_), (None, None, []))
            files = [f for f in files if f.endswith(".png")]

            n_images = len(files)
            img_shuffle = np.random.permutation(n_images)

            for i, v in enumerate(files):
                img_name = ('%05i.png' % img_shuffle[i])
                shutil.copyfile(os.path.join(dir_, v),
                                os.path.join(dir_real, img_name))


def generate_physics_data(datadir):
    """
    generate_physics_data(datadir)

    Generates data to use as the physics-informed image pairs

    This data is stored in a ./paired_data/ subfolder of <datadir>

    Parameters
    ----------
    datadir : String,
        Path to folder containing PhysicsDataConfig.yml config file

    Returns
    -------
    None.

    """

    config, psfconfigLR, psfconfigHR = load_config(datadir + '/PhysicsDataConfig.yml')
    if 'real' in config['data']:
        # Load real ground truth and simulate a lower resolution
        simulate_lsm_convolution(config, psfconfigLR, psfconfigHR=None, data_mode='real')
    if 'sim' in config['data']:
        # Simulate HR and LR images of beads and speckle
        simulate_lsm_vectorial(config, psfconfigLR, psfconfigHR)
        simulate_lsm_convolution(config, psfconfigLR, psfconfigHR, data_mode='sim')
        # simulate_lsm_noise(config)


# !DEPRECIATED
def prepare_sim_data(datadir):
    config, psfconfigLR, psfconfigHR = load_config(datadir + '/TrainConfig.yml')

    if 'real' in config['data']:
        simulate_lsm_convolution(config, psfconfigLR, psfconfigHR=None, data_mode='real')
    if 'sim' in config['data']:
        simulate_lsm_vectorial(config, psfconfigLR, psfconfigHR)
        simulate_lsm_convolution(config, psfconfigLR, psfconfigHR, data_mode='sim')

    rootdir = os.path.join(config['dir'], config['date'])
    return rootdir


# !DEPRECIATED
def prepare_real_data(datadir):
    config, psfconfigLR, psfconfigHR = load_config(datadir + '/TrainConfig.yml')
    rootdir = os.path.join(config['dir'], config['date'])
    realdir = os.path.join(rootdir, 'real')
    realsource = config['data_real_dir']

    (_, _, files) = next(os.walk(realsource), (None, None, []))
    files = [f for f in files if f.endswith(".png")]

    os.makedirs(realdir, exist_ok=True)

    for i, v in enumerate(files):
        shutil.copyfile(os.path.join(realsource, v), os.path.join(realdir, v))
