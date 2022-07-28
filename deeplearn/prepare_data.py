# -*- coding: utf-8 -*-
"""Prepare data for training based on a specified config file
"""

# Created on Wed June 17 2020

# Contributions and Changelog accessed via a private git repository:
# https://github.com/philipwijesinghe/deep-learning-lsm.git
# philip.wijesinghe@gmail.com

# Copyright (c) 2020 Philip Wijesinghe@University of St Andrews (pw64@st-andrews.ac.uk)

import sys
sys.path.append('../')

import os
import shutil
import glob
import numpy as np

from skimage.transform import resize
from PIL import Image

from fileio.yamlio import load_config
from fileio.imageio import load_img_stack


# for repeatability, set RNG seed
# np.random.seed(10)


# =============================================================================
# FUNCTIONS
# =============================================================================
def prepare_training_data(datadir, overwrite=True, **kwargs):
    """Populates folder for network training with data specified in TrainConfig.yml

    Auto separates training and validation data

    Parameters
    ----------
    datadir : String
        Path to folder containing TrainConfig.yml
    overwrite
        True (default) | False : if true, will delete existing images in the directory
    kwargs
    (Optional)
        data='sim' if no real data is to be used for training

    Returns
    -------
    None

    """
    train_real = True
    if 'data' in kwargs:
        if kwargs['data'] == 'sim':
            train_real = False

    config, _, _ = load_config(datadir + '/TrainConfig.yml')

    # Parse location of raw data
    dirs_phys_data_raw = config['phys_dirs'].split()
    dirs_real_data_raw = config['real_dirs'].split() if train_real else None

    # Prepare folders for training
    rootdir = config['dir']
    dir_train = os.path.join(rootdir, 'train')
    dir_val = os.path.join(rootdir, 'val')
    dir_real = os.path.join(rootdir, 'real') if train_real else None

    # important to not have old data present
    if os.path.exists(dir_train):
        if overwrite:
            shutil.rmtree(dir_train)
            shutil.rmtree(dir_val)
            if train_real:
                shutil.rmtree(dir_real)
        else:
            print('Folders already exist - please remove')
            return
    try:
        os.makedirs(dir_train, exist_ok=False)
        os.makedirs(dir_val, exist_ok=False)
        if train_real:
            os.makedirs(dir_real, exist_ok=False)
    except Exception as e:
        print('Error in making directories. See exception: ')
        print(e)
        return

    # Count total phys images
    n_images_phys = 0
    for dir_ in dirs_phys_data_raw:
        # TODO: support for non-png images (support in dataloader first)
        file_list = [f for f in glob.glob(dir_ + "\\*.png")]
        n_images_phys += len(file_list)

    # Validation shuffler
    n_val = config['n_val']
    rperm = np.random.permutation(n_images_phys)
    rperm = rperm[:n_val]

    # Copy phys images
    n_offset = 0
    for dir_ in dirs_phys_data_raw:
        (_, _, files) = next(os.walk(dir_), (None, None, []))
        files = [f for f in files if f.endswith(".png")]

        n_images = len(files)
        img_shuffle = np.random.permutation(n_images) + n_offset

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
        for dir_ in dirs_real_data_raw:
            (_, _, files) = next(os.walk(dir_), (None, None, []))
            files = [f for f in files if f.endswith(".png")]

            n_images = len(files)
            img_shuffle = np.random.permutation(n_images)

            for i, v in enumerate(files):
                img_name = ('%05i.png' % img_shuffle[i])
                shutil.copyfile(os.path.join(dir_, v),
                                os.path.join(dir_real, img_name))


def generate_physics_data(datadir):
    """Simulates physics data to use as the physics-informed image pairs

    Simulation is configured using the PhysicsDataConfig.yml file, which must be present in the datadir

    This paired data will be stored in a ./paired_data/ subfolder of <datadir>

    Parameters
    ----------
    datadir : String,
        Path to folder containing PhysicsDataConfig.yml config file

    Returns
    -------
    None.

    """

    from lsm.simulate import simulate_lsm_real, simulate_lsm_beads, simulate_lsm_speckle

    config, psfconfig_lr, psfconfig_hr = load_config(datadir + '/PhysicsDataConfig.yml')

    if 'real' in config['data']:
        # Load real ground truth and simulate a lower resolution
        simulate_lsm_real(config, psfconfig_lr)

    if 'sim' in config['data']:
        if 'bead' in config['sim']['type']:
            simulate_lsm_beads(config, psfconfig_lr, psfconfig_hr)

        if 'speckle' in config['sim']['type']:
            simulate_lsm_speckle(config, psfconfig_lr, psfconfig_hr)


def prepare_real_data(datadir, outdir, **kwargs):
    """Crops image stacks into small ROIs for deep learning training

    Parameters
    ----------
    datadir
        location of image stack - must be a sequence of images (vertical: z; horisontal: x)
    outdir
        location where to store output images
    kwargs
        out_size = 64
            size in pix of the output images
        in_pix_size = 1
            pixel size of input image
        out_pix_size = 1
            pixel size of output image
        focus_centre = image_size_z // 2
            index position in z where the focus is (ROIs will be selected from there)
        crop_type = 'centre' | 'random' | 'brightest'
            'centre' will select the central ROI of each image
            'random' will select a random ROI for each image around the focus
            'brightest' will select a random ROI around the brightest area in each image
        normalise = True | False
            normalise the output images to min max values
        keep_percentile = 50
            keeps only the images with intensity > percentile (0 will keep all)
    """

    img_stack = load_img_stack(datadir)
    img_size_izx = img_stack.shape

    crop_type = kwargs['crop_type'] if 'crop_type' in kwargs else 'centre'
    keep_percentile = kwargs['keep_percentile'] if 'keep_percentile' in kwargs else 50
    normalise = kwargs['normalise'] if 'normalise' in kwargs else True
    out_size = kwargs['out_size'] if 'out_size' in kwargs else 64
    in_pix_size = kwargs['in_pix_size'] if 'in_pix_size' in kwargs else 1
    out_pix_size = kwargs['out_pix_size'] if 'out_pix_size' in kwargs else 1
    focus_centre = kwargs['focus_centre'] if 'focus_centre' in kwargs else img_size_izx[1] // 2

    out_fov = out_size * out_pix_size

    # first we need to rescale input image to output pixel size
    if in_pix_size != out_pix_size:
        scale = in_pix_size / out_pix_size
        new_img_size_izx = scale * img_size_izx
        new_img_size_izx[0] = img_size_izx[0]
        img_stack = resize(img_stack, new_img_size_izx)
        img_size_izx = img_stack.shape
        focus_centre = int(scale * focus_centre)

    # crop the z to the appropriate focus region
    z_range = min(2 * out_fov, img_size_izx[1])
    if z_range < out_fov:
        print('Error, the input images are small than the desired crop region.')
        return

    z_vect = np.arange(z_range) - z_range // 2 + focus_centre
    if np.min(z_vect) < 0:
        print('Focus centre too close to top surface')
        return
    if np.max(z_vect) > img_size_izx[1] - 1:
        print('Focus centre too close to bottom surface')
        return

    img_stack = img_stack[:, z_vect, :]
    img_size_izx = img_stack.shape

    # select and crop regions
    img_cropped = np.zeros([img_size_izx[0], out_size, out_size])
    for i_ in range(img_size_izx[0]):
        img_single = img_stack[i_, :, :]

        crop_vector = np.arange(out_size)

        if crop_type == 'centre':
            crop_z = crop_vector + img_size_izx[1] // 2 - out_size // 2
            crop_x = crop_vector + img_size_izx[2] // 2 - out_size // 2
        elif crop_type == 'random':
            crop_z = crop_vector + np.random.randint(0, img_size_izx[1] - out_size)
            crop_x = crop_vector + np.random.randint(0, img_size_izx[2] - out_size)
        elif crop_type == 'brightest':
            # find and centre at brightest point in x
            x_max_vect = np.max(img_single, axis=0)
            x_max_vect = np.convolve(x_max_vect, np.ones([out_size // 2, ]), mode='same')
            x_max_pos = np.argmax(x_max_vect)
            x_max_pos = np.maximum(x_max_pos, out_size // 2 + 1)
            x_max_pos = np.minimum(x_max_pos, img_size_izx[2] - out_size // 2 - 1)
            crop_x = crop_vector + x_max_pos - out_size // 2
            crop_z = crop_vector + np.random.randint(0, img_size_izx[1] - out_size)
        else:
            print('Unrecognised crop type')
            return

        tmp = img_single[crop_z, :]
        tmp = tmp[:, crop_x]
        img_cropped[i_, :, :] = tmp

    # select the brightest %
    mean_intensity = np.mean(img_cropped, axis=(1, 2))
    index_keep = mean_intensity > np.percentile(mean_intensity, keep_percentile)
    img_cropped = img_cropped[index_keep, :, :]

    # normalise
    if normalise:
        img_cropped = img_cropped.astype('float') - np.min(img_cropped)
        img_cropped = img_cropped / np.max(img_cropped)
        img_cropped = img_cropped * 0.8 + 0.1  # soft crop to avoid loss clipping
        img_cropped = img_cropped * 255

    # save images
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    for i_ in range(img_cropped.shape[0]):
        img_save = img_cropped[i_, :, :]
        img = Image.new('L', (out_size, out_size))
        img.paste(Image.fromarray(img_save), (0, 0))
        img.save(outdir + '\\' + '%05i.png' % i_)
