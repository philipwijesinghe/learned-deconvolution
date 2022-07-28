# -*- coding: utf-8 -*-
"""PSF module

Generates an LSM point-spread function
"""

# Created on Wed Jun  17, 2020

# Contributions and Changelog accessed via a private git repository:
# https://github.com/philipwijesinghe/deep-learning-lsm.git
# philip.wijesinghe@gmail.com

# Copyright (c) 2020 Philip Wijesinghe@University of St Andrews (pw64@st-andrews.ac.uk)

import re
import os
import glob
import numpy as np

from PIL import Image
from scipy.signal import convolve2d
from skimage.util import random_noise
from skimage import io

from lsm.psf import gen_psf_vect
from fileio.imageio import save_image

from joblib import Parallel, delayed


# =============================================================================
# FUNCTIONS
# =============================================================================
def scattering_psf(xv, zv, psfconfig, psfdistmax, y=None):
    """
    Calculates the detected psf of a single scatterer located at (x,z) = (0,0)
    evaluated at a set of coordinates given in xv and zv.

    Parameters
    ----------
    xv : indexable array
        vector of x coordinates.
    zv : indexable array
        vector of z coordinates.
    psfconfig : psfConfig Object (lsm.psf module)
        object describing the light-sheet PSF.
    psfdistmax : float
        maximum size of the psf; coordinates beyond this radius are not
        evaluated for speed; in meters.
    (Optional)
    y : indexable array
        vector of y defocus of the scatterer for 3D projection simulation.

    Returns
    -------
    scattering_signal : 1D numpy array
        intensity of the field defined by (xv,zv) due to a scatterer at (0,0)

    """
    # tic = time.perf_counter()
    scattering_signal = np.zeros(xv.shape)
    if y is None:
        for pi in range(len(xv)):
            if xv[pi] ** 2 + zv[pi] ** 2 < psfdistmax ** 2:
                scattering_signal[pi] = gen_psf_vect(np.array([xv[pi]]), np.array([zv[pi]]),
                                                     lambda0=488e-9, psfconfig=psfconfig)
    else:
        for pi in range(len(xv)):
            if xv[pi] ** 2 + zv[pi] ** 2 + y[pi] ** 2 < psfdistmax ** 2:
                scattering_signal[pi] = gen_psf_vect(np.array([xv[pi]]), np.array([zv[pi]]), y=np.array([y[pi]]),
                                                     lambda0=488e-9, psfconfig=psfconfig)

    # TODO: add option for specifying lambda!!!

    # toc = time.perf_counter()
    # print(f"Method 1 {toc - tic:0.4f} seconds")
    return scattering_signal


def gen_psf(config, psfconfig_lr, psfconfig_hr=None):
    """Generates a PSF image from configuration

    Returns PSF image arrays for LR config and HR config (if specified)

    Parameters
    ----------
    config
        config dictionary from load_config of PhysicsDataConfig.yml file
    psfconfig_lr
        LR PSF configuration using PSFConfig Class
    psfconfig_hr
        HR PSF configuration using PSFConfig Class

    Returns
    -------
    psf_lr, psf_hr
        numpy 2D arrays (images of the PSFs)

    """
    # Generate psf images for convolution
    sim_conf = config['sim']
    noise_conf = config['noise']

    npixpsf = int(sim_conf['psfDistMaxLR'] // config['pix_spacing'])
    xp = np.linspace(-npixpsf * config['pix_spacing'],
                     npixpsf * config['pix_spacing'],
                     npixpsf * 2 + 1)
    xpr, zpr = np.meshgrid(xp, xp, sparse=False, indexing='ij')

    # rotation via rotation matrix
    pxvr, pzvr = rotate_psf(
        xpr, zpr, config['theta'], theta_var=noise_conf['theta_var']
    )
    # linearize vectors
    pxvr = pxvr.flatten()
    pzvr = pzvr.flatten()

    if psfconfig_lr.config['experimental']['dir']:
        psf_lr = io.imread(psfconfig_lr.config['experimental']['dir'])
        psf_lr = np.transpose(psf_lr).astype('double')
        psf_lr -= np.min(psf_lr)
        psf_lr = psf_lr / np.max(psf_lr)
    else:
        psf_lr = scattering_psf(pxvr, pzvr, psfconfig_lr, sim_conf['psfDistMaxLR'])
        psf_lr = psf_lr.reshape((npixpsf * 2 + 1, npixpsf * 2 + 1))
    # only supports experimental in LR at the moment
    # TODO: Fix all this with proper PSF classes

    if psfconfig_hr:
        psf_hr = scattering_psf(pxvr, pzvr, psfconfig_hr, sim_conf['psfDistMaxHR'])
        psf_hr = psf_hr.reshape((npixpsf * 2 + 1, npixpsf * 2 + 1))
    else:
        psf_hr = None

    return psf_lr, psf_hr


def gen_psf_3d(config, psfconfig_lr, psfconfig_hr=None):
    """Generates a 3D psf from configuration

    Returns PSF image arrays for LR config and HR config (if specified)

    Parameters
    ----------
    config
        config dictionary from load_config of PhysicsDataConfig.yml file
    psfconfig_lr
        LR PSF configuration using PSFConfig Class
    psfconfig_hr
        HR PSF configuration using PSFConfig Class

    Returns
    -------
    psf_lr, psf_hr
        numpy 3D arrays

    """
    if psfconfig_lr.config['experimental']['dir']:
        print('3D psf is not supported for experimental PSF images')
        return

    # Generate psf images for convolution
    sim_conf = config['sim']
    noise_conf = config['noise']

    # Create PSFs
    npixpsf = int(sim_conf['psfDistMaxLR'] // config['pix_spacing'])
    xp = np.linspace(-npixpsf * config['pix_spacing'],
                     npixpsf * config['pix_spacing'],
                     npixpsf * 2 + 1)
    xpr, zpr = np.meshgrid(xp, xp, sparse=False, indexing='ij')

    # rotation via rotation matrix
    pxvr, pzvr = rotate_psf(
        xpr, zpr, config['theta'], theta_var=noise_conf['theta_var']
    )

    # linearize vectors
    pxvr = pxvr.flatten()
    pzvr = pzvr.flatten()

    yp = xp
    psf_lr = np.zeros([npixpsf * 2 + 1, npixpsf * 2 + 1, npixpsf * 2 + 1])
    for yi in range(len(yp)):
        yvr = np.ones(pzvr.shape) * yp[yi]
        psf_lr2d = scattering_psf(pxvr, pzvr, psfconfig_lr, sim_conf['psfDistMaxLR'], y=yvr)
        psf_lr2d = psf_lr2d.reshape((npixpsf * 2 + 1, npixpsf * 2 + 1))
        psf_lr[:, :, yi] = psf_lr2d

    if psfconfig_hr:
        psf_hr = scattering_psf(pxvr, pzvr, psfconfig_hr, sim_conf['psfDistMaxHR'])
        psf_hr = psf_hr.reshape((npixpsf * 2 + 1, npixpsf * 2 + 1))
    else:
        psf_hr = None

    return psf_lr, psf_hr


def add_noise(img, noise_conf):
    """Adds noise to image using skimage tools based on config

    Parameters
    ----------
    img
        numpy 2D array
    noise_conf
        noise configuration dict loaded from yml

    Returns
    -------
    numpy 2D array

    """
    def rnd():
        return np.random.normal(1, noise_conf['noise_var'])

    if 'gaussian' in noise_conf['type']:
        img = random_noise(
            img,
            mode='gaussian',
            mean=noise_conf['gaussian_mu'] * rnd(),
            var=(noise_conf['gaussian_std'] * rnd()) ** 2
        )
    if 'speckle' in noise_conf['type']:
        img = random_noise(
            img,
            mode='speckle',
            mean=noise_conf['gaussian_mu'] * rnd(),
            var=(noise_conf['gaussian_std'] * rnd()) ** 2
        )
    if 'snp' in noise_conf['type']:
        img = random_noise(
            img,
            mode='s&p',
            amount=noise_conf['saltpepper'] * rnd()
        )

    return img


def add_normalise_noise(img, noise_conf):
    """Normalises image and adds noise based on config

    Parameters
    ----------
    img
        numpy 2D array
    noise_conf
        noise configuration dict loaded from yml

    Returns
    -------
    numpy 2D array

    """
    # normalise [0, 1]
    img -= np.min(img)
    img = img / np.max(img)

    # buffer for normalisation [0.1 0.9] (This keeps image values consistent for quant. compraisons later)
    img = img * 0.8 + 0.1

    # add noise
    if noise_conf:
        img = add_noise(img, noise_conf)

    # clip and renormalise to [0 255] (will be clipped on saving to uint8)
    img = img * 255

    return img


def rotate_psf(xv, zv, theta, theta_var=None):
    """Rotate coordinate papers"""
    # randomization
    if theta_var:
        theta += 2 * np.pi * theta_var * np.random.randn()

    # rotation via rotation matrix
    xvr = np.cos(theta) * xv - np.sin(theta) * zv
    zvr = np.sin(theta) * xv + np.cos(theta) * zv

    return xvr, zvr


def find_last_image_no(dir_):
    """finds the last numerical value in a set of filenames"""
    file_list = [os.path.basename(f) for f in glob.glob(dir_ + "\\*.png")]

    def extract_number(f):
        s = re.findall(r"\d+", f)
        return int(s[-1]) if s else -1

    return extract_number(max(file_list, key=extract_number)) if file_list else -1


def create_dirs(config):
    dir_ = config['dir'] + '\\paired_data'
    os.makedirs(dir_, exist_ok=True)
    return dir_


# =============================================================================
# SIMULATE LR / HR PAIR IMAGES
# =============================================================================
def simulate_lsm_beads(config, psfconfig_lr, psfconfig_hr, parallel=True):
    """Simulates LR / HR image pairs of beads (impulse scatterers)

    Parameters
    ----------
    config
        config dictionary from load_config of PhysicsDataConfig.yml file
    psfconfig_lr
        LR PSF configuration using PSFConfig Class
    psfconfig_hr
        HR PSF configuration using PSFConfig Class
    (Optional)
    parallel
        True | False - use parallel threading
    """
    # Prepare directory
    dir_ = create_dirs(config)
    last_n = find_last_image_no(dir_)

    # Input formatting
    sim_conf = config['sim']
    noise_conf = config['noise']
    fov_xz = np.array(config['img_size_xz']) * config['pix_spacing']

    if 'speckle' in sim_conf['type']:
        n_simulations = int(sim_conf['n_images'] * sim_conf['sparsity_split'])
    else:
        n_simulations = int(sim_conf['n_images'])

    # Image coordinates
    x = np.linspace(0, fov_xz[0], config['img_size_xz'][0])
    z = np.linspace(0, fov_xz[1], config['img_size_xz'][1])

    psf_lr, psf_hr = gen_psf(config, psfconfig_lr, psfconfig_hr)

    # If an experimental psf is used, then use convolution for LR image simulation
    experimental = True if psfconfig_lr.config['experimental']['dir'] else False

    # for ni in range(sim_conf['n_images']):
    def sim_single(ni):
        # Let's create a random distribution of scatterers
        no_scatterers = np.random.randint(
            sim_conf['n_scatterers_min_max'][0],
            sim_conf['n_scatterers_min_max'][1]
        )
        scatterer_locs_x = np.random.rand(no_scatterers) * fov_xz[0]
        scatterer_locs_z = np.random.rand(no_scatterers) * fov_xz[1]
        scatterer_intensity = np.random.rand(no_scatterers)

        # Simulate single image as a sum of spatially shifted impulse responses
        # Uses vectorised code for subpixel accuracy
        image_lr = np.zeros(config['img_size_xz'])
        image_hr = np.zeros(config['img_size_xz'])
        for si in range(no_scatterers):
            # shift coordinates relative to scatterer position
            sx = x - scatterer_locs_x[si]
            sz = z - scatterer_locs_z[si]

            # create coordinate vector field
            sxv, szv = np.meshgrid(sx, sz, sparse=False, indexing='ij')

            # rotation via rotation matrix
            sxvr, szvr = rotate_psf(
                sxv, szv, config['theta'], theta_var=noise_conf['theta_var']
            )

            # linearize vectors
            sxvr = sxvr.flatten()
            szvr = szvr.flatten()

            # High resolution image
            # =====================================================================
            scattering_signal = \
                scatterer_intensity[si] * scattering_psf(sxvr, szvr, psfconfig_hr, sim_conf['psfDistMaxHR'])
            image_hr += scattering_signal.reshape(config['img_size_xz'])

            # Low resolution image
            # =====================================================================
            if not experimental:
                scattering_signal = \
                    scatterer_intensity[si] * scattering_psf(sxvr, szvr, psfconfig_lr, sim_conf['psfDistMaxLR'])
                image_lr += scattering_signal.reshape(config['img_size_xz'])

        if experimental:
            image_lr = convolve2d(image_hr, psf_lr, mode='same', boundary='fill')

        # Format output noise
        image_lr = add_normalise_noise(image_lr, noise_conf)
        image_hr = add_normalise_noise(
            image_hr,
            noise_conf if noise_conf['add_noise_to_HR'] else 0
        )

        # Concatenate and save as images
        save_image(config['dir'], None, last_n + 1 + ni,
                   imageLR_xz=image_lr, imageHR_xz=image_hr, mode='paired')

        print("Generated %i of %i images" % (ni + 1, n_simulations))

    if parallel:
        Parallel(n_jobs=4)(delayed(sim_single)(ni) for ni in range(n_simulations))
    else:
        for ni in range(sim_conf['n_images']):
            sim_single(ni)


def simulate_lsm_speckle(config, psfconfig_lr, psfconfig_hr):
    """Simulated LR / HR image pairs of speckle

    Parameters
    ----------
    config
        config dictionary from load_config of PhysicsDataConfig.yml file
    psfconfig_lr
        LR PSF configuration using PSFConfig Class
    psfconfig_hr
        HR PSF configuration using PSFConfig Class

    Returns
    -------

    """
    dir_ = create_dirs(config)
    last_n = find_last_image_no(dir_)

    sim_conf = config['sim']
    noise_conf = config['noise']

    if 'bead' in sim_conf['type']:
        n_simulations = int(sim_conf['n_images']) - int(sim_conf['n_images'] * sim_conf['sparsity_split'])
    else:
        n_simulations = int(sim_conf['n_images'])

    psf_lr, psf_hr = gen_psf(config, psfconfig_lr, psfconfig_hr)
    for ni in range(n_simulations):
        # Create LR and HR images via convolution
        xv = np.linspace(-config['img_size_xz'][0],
                         config['img_size_xz'][0],
                         1 * config['img_size_xz'][0]) / 2
        x, y = np.meshgrid(xv, xv)

        # Speckle is generated as a fft of a random phase apodised with low and high frequency Gaussian filters
        hf_speckle = np.exp(-(x ** 2 + y ** 2) * sim_conf['speckle']['hf_size_pix'] ** 2)
        lf_speckle = np.exp(-(x ** 2 + y ** 2) * sim_conf['speckle']['lf_size_pix'] ** 2)
        rimg = ((hf_speckle + sim_conf['speckle']['weight'] * lf_speckle) *
                np.exp(-1j * 2 * np.pi * np.random.rand(x.shape[0], x.shape[1])))
        rimg = np.fft.fftshift(np.fft.fft2(rimg, s=x.shape))
        rimg = np.abs(rimg) ** 2

        image_lr = convolve2d(rimg, psf_lr, mode='same', boundary='fill')
        image_hr = convolve2d(rimg, psf_hr, mode='same', boundary='fill')

        # Normalize
        image_lr = add_normalise_noise(image_lr, noise_conf)
        image_hr = add_normalise_noise(
            image_hr,
            noise_conf if noise_conf['add_noise_to_HR'] else 0
        )

        save_image(config['dir'], None, last_n + 1 + ni,
                   imageLR_xz=image_lr, imageHR_xz=image_hr, mode='paired')

        print("Generated %i of %i images" % (ni + 1, n_simulations))


def simulate_lsm_real(config, psfconfig_lr):
    """Simulates LSM images from real HR images

    Parameters
    ----------
    config
        config dictionary from load_config of PhysicsDataConfig.yml file
    psfconfig_lr
        LR PSF configuration using PSFConfig Class

    Returns
    -------

    """
    dir_ = create_dirs(config)
    last_n = find_last_image_no(dir_)

    sim_conf = config['sim']
    noise_conf = config['noise']

    file_list = [f for f in glob.glob(config['data_dir'] + "\\*.png")]
    file_list += [f for f in glob.glob(config['data_dir'] + "\\*.tif")]
    n_images = len(file_list)
    # n_val = config['n_val_real']
    # rperm = np.random.permutation(n_images)
    # rperm = rperm[:n_val]

    psf_lr, _ = gen_psf(config, psfconfig_lr)

    for ni in range(n_images):
        # Create LR and HR images via convolution

        # read image
        rimg = Image.open(file_list[ni]).convert('I')
        # TODO: manage multiple input formats; currently uint16->int32
        image_hr = np.transpose(np.float64(np.asarray(rimg)))
        image_lr = convolve2d(image_hr, psf_lr, mode='same', boundary='fill')

        # Normalize
        image_lr = add_normalise_noise(image_lr, noise_conf)

        save_image(config['dir'], None, last_n + 1 + ni,
                   imageLR_xz=image_lr, imageHR_xz=image_hr, mode='paired')

        print("Generated %i of %i images" % (ni + 1, n_images))


# =============================================================================
# !DEPRECATED CODE BELOW
# =============================================================================
def simulate_lsm_vectorial(config, psfconfigLR, psfconfigHR=None):
    print("WARNING: USING DEPRECATED CODE")

    # Prepare directory
    if psfconfigHR:
        # dir_ = config['dir']+'\\'+config['date']+'\\train'
        # dir_val = config['dir']+'\\'+config['date']+'\\val'
        # dir_ref = config['dir']+'\\'+config['date']+'\\reference'
        # # mode = 'train'
        # os.makedirs(dir_val, exist_ok=True)
        # os.makedirs(dir_ref, exist_ok=True)
        dir_ = create_dirs(config)
    else:
        dir_ = config['dir'] + '\\' + config['date'] + '\\process'
        os.makedirs(dir_, exist_ok=True)

    last_n = find_last_image_no(dir_)

    # Input formatting
    sim_conf = config['sim']
    noise_conf = config['noise']
    fov_xz = np.array(config['img_size_xz']) * config['pix_spacing']

    # Image coordinates
    x = np.linspace(0, fov_xz[0], config['img_size_xz'][0])
    z = np.linspace(0, fov_xz[1], config['img_size_xz'][1])

    for ni in range(sim_conf['n_images']):
        # Let's create a random distribution of scatterers
        no_scatterers = np.random.randint(
            sim_conf['n_scatterers_min_max'][0],
            sim_conf['n_scatterers_min_max'][1]
        )
        scatterer_locs_x = np.random.rand(no_scatterers) * fov_xz[0]
        scatterer_locs_z = np.random.rand(no_scatterers) * fov_xz[1]
        scatterer_intensity = np.random.rand(no_scatterers)

        # Simulate LSM
        imageLR = np.zeros(config['img_size_xz'])
        imageHR = np.zeros(config['img_size_xz'])
        for si in range(no_scatterers):
            # shift coordinates relative to scatterer
            sx = x - scatterer_locs_x[si]
            sz = z - scatterer_locs_z[si]

            # create coordinate vector field
            sxv, szv = np.meshgrid(sx, sz, sparse=False, indexing='ij')

            # rotation via rotation matrix
            sxvr, szvr = rotate_psf(
                sxv, szv, config['theta'], theta_var=noise_conf['theta_var']
            )

            # linearize vectors
            sxvr = sxvr.flatten()
            szvr = szvr.flatten()

            # Low resolution image
            # =====================================================================
            scattering_signal = (
                    scatterer_intensity[si] * scattering_psf(sxvr, szvr, psfconfigLR,
                                                             sim_conf['psfDistMaxLR'])
            )
            imageLR += scattering_signal.reshape(config['img_size_xz'])

            # High resolution image
            # =====================================================================
            if psfconfigHR:
                scattering_signal = (
                        scatterer_intensity[si] * scattering_psf(sxvr, szvr, psfconfigHR,
                                                                 sim_conf['psfDistMaxHR'])
                )
                imageHR += scattering_signal.reshape(config['img_size_xz'])

        # Format output noise
        imageLR = add_normalise_noise(imageLR, noise_conf)

        if psfconfigHR:
            imageHR = add_normalise_noise(
                imageHR,
                noise_conf if noise_conf['add_noise_to_HR'] else 0
            )

            # save_image(config['dir'], config['date'], last_n+1+ni,
            #            imageLR_xz=imageHR, imageHR_xz=None, mode='reference')

            # if noise_conf['add_noise_to_HR']:
            #     imageHR = add_normalise_noise(imageHR, noise_conf)

        # Concatenate and save as images
        if psfconfigHR:
            # if ni < sim_conf['n_val']:
            #     save_image(config['dir'], config['date'], last_n+1+ni,
            #                imageLR_xz=imageLR, imageHR_xz=imageHR, mode='val')
            # else:
            #     save_image(config['dir'], config['date'], last_n+1+ni,
            #                imageLR_xz=imageLR, imageHR_xz=imageHR, mode='train')
            save_image(config['dir'], None, last_n + 1 + ni,
                       imageLR_xz=imageLR, imageHR_xz=imageHR, mode='paired')
        else:
            save_image(config['dir'], config['date'], last_n + 1 + ni,
                       imageLR_xz=imageLR, imageHR_xz=None, mode='process')

        print("Generated %i of %i images" % (ni + 1, sim_conf['n_images']))


def simulate_lsm_lr_convolution(config, psfconfigLR, psfconfigHR):
    # Prepare directory
    dir_ = create_dirs(config)
    last_n = find_last_image_no(dir_)

    # Input formatting
    sim_conf = config['sim']
    noise_conf = config['noise']
    fov_xz = np.array(config['img_size_xz']) * config['pix_spacing']

    # Image coordinates
    x = np.linspace(0, fov_xz[0], config['img_size_xz'][0])
    z = np.linspace(0, fov_xz[1], config['img_size_xz'][1])

    psfLR, psfHR = gen_psf(config, psfconfigLR, psfconfigHR)

    # Try parallel
    # for ni in range(sim_conf['n_images']):
    def sim_single(ni):
        # Let's create a random distribution of scatterers
        no_scatterers = np.random.randint(
            sim_conf['n_scatterers_min_max'][0],
            sim_conf['n_scatterers_min_max'][1]
        )
        scatterer_locs_x = np.random.rand(no_scatterers) * fov_xz[0]
        scatterer_locs_z = np.random.rand(no_scatterers) * fov_xz[1]
        scatterer_intensity = np.random.rand(no_scatterers)

        # Simulate LSM
        imageLR = np.zeros(config['img_size_xz'])
        imageHR = np.zeros(config['img_size_xz'])
        for si in range(no_scatterers):
            # shift coordinates relative to scatterer
            sx = x - scatterer_locs_x[si]
            sz = z - scatterer_locs_z[si]

            # create coordinate vector field
            sxv, szv = np.meshgrid(sx, sz, sparse=False, indexing='ij')

            # rotation via rotation matrix
            sxvr, szvr = rotate_psf(
                sxv, szv, config['theta'], theta_var=noise_conf['theta_var']
            )

            # linearize vectors
            sxvr = sxvr.flatten()
            szvr = szvr.flatten()

            # High resolution image
            # =====================================================================
            scattering_signal = (
                    scatterer_intensity[si] * scattering_psf(sxvr, szvr, psfconfigHR,
                                                             sim_conf['psfDistMaxHR'])
            )
            imageHR += scattering_signal.reshape(config['img_size_xz'])

            # Low resolution image
            # =====================================================================
            imageLR = convolve2d(imageHR, psfLR, mode='same', boundary='fill')

        # Format output noise
        imageLR = add_normalise_noise(imageLR, noise_conf)

        if psfconfigHR:
            imageHR = add_normalise_noise(
                imageHR,
                noise_conf if noise_conf['add_noise_to_HR'] else 0
            )

            # save_image(config['dir'], config['date'], last_n+1+ni,
            #            imageLR_xz=imageHR, imageHR_xz=None, mode='reference')

            # if noise_conf['add_noise_to_HR']:
            #     imageHR = add_normalise_noise(imageHR, noise_conf)

        # Concatenate and save as images
        if psfconfigHR:
            # if ni < sim_conf['n_val']:
            #     save_image(config['dir'], config['date'], last_n+1+ni,
            #                imageLR_xz=imageLR, imageHR_xz=imageHR, mode='val')
            # else:
            #     save_image(config['dir'], config['date'], last_n+1+ni,
            #                imageLR_xz=imageLR, imageHR_xz=imageHR, mode='train')
            save_image(config['dir'], None, last_n + 1 + ni,
                       imageLR_xz=imageLR, imageHR_xz=imageHR, mode='paired')
        else:
            save_image(config['dir'], config['date'], last_n + 1 + ni,
                       imageLR_xz=imageLR, imageHR_xz=None, mode='process')

        print("Generated %i of %i images" % (ni + 1, sim_conf['n_images']))

    Parallel(n_jobs=6)(delayed(sim_single)(ni) for ni in range(sim_conf['n_images']))


def simulate_lsm_convolution(config, psfconfigLR, psfconfigHR=None,
                             data_mode='sim', save_mode='train'):
    # if sim then randomly make an image
    # if real then read and convolve from a folder of hr images
    if psfconfigHR:
        dir_ = create_dirs(config)
    else:
        dir_ = create_dirs(config)
        # dir_ = config['dir'] + '\\' + config['date'] + '\\process'
        # os.makedirs(dir_, exist_ok=True)

    last_n = find_last_image_no(dir_)

    sim_conf = config['sim']
    noise_conf = config['noise']

    if data_mode == 'sim':
        n_images = sim_conf['n_images']
        n_val = sim_conf['n_val']
        rperm = range(n_val)
    elif data_mode == 'real':
        file_list = [f for f in glob.glob(config['data_dir'] + "\\*.png")]
        file_list += [f for f in glob.glob(config['data_dir'] + "\\*.tif")]
        n_images = len(file_list)
        n_val = n_images  # config['n_val_real']
        rperm = np.random.permutation(n_images)
        rperm = rperm[:n_val]
        # TODO: LIBTIFF Error - critical - cannot load TIF under windows
    else:
        print("Incorrect data_mode specified. Must be 'sim' or 'real'.")
        return

    for ni in range(n_images):
        # Generate PSF
        psfLR, psfHR = gen_psf(config, psfconfigLR, psfconfigHR)

        # Create LR and HR images via convolution
        if data_mode == 'sim':
            # rimg = np.random.rand(config['img_size_xz'][0], config['img_size_xz'][1])

            x = np.linspace(-config['img_size_xz'][0],
                            config['img_size_xz'][0],
                            1 * config['img_size_xz'][0]) / 2
            X, Y = np.meshgrid(x, x)

            hf_speckle = np.exp(-(X ** 2 + Y ** 2) * sim_conf['speckle']['hf_size_pix'] ** 2)
            lf_speckle = np.exp(-(X ** 2 + Y ** 2) * sim_conf['speckle']['lf_size_pix'] ** 2)
            U = ((hf_speckle + sim_conf['speckle']['weight'] * lf_speckle) *
                 np.exp(-1j * 2 * np.pi * np.random.rand(X.shape[0], X.shape[1])))

            rimg = np.fft.fftshift(np.fft.fft2(U, s=X.shape))
            rimg = np.abs(rimg) ** 2

            imageLR = convolve2d(rimg, psfLR, mode='same', boundary='fill')
            if psfconfigHR:
                imageHR = convolve2d(rimg, psfHR, mode='same', boundary='fill')
            else:
                imageHR = None
        elif data_mode == 'real':
            # read image
            rimg = Image.open(file_list[ni]).convert('I')
            # TODO: manage multiple input formats; currently uint16->int32
            imageHR = np.transpose(np.float64(np.asarray(rimg)))
            imageLR = convolve2d(imageHR, psfLR, mode='same', boundary='fill')

        # Normalize
        imageLR = add_normalise_noise(imageLR,
                                      noise_conf)

        if isinstance(imageHR, np.ndarray):
            imageHR = add_normalise_noise(
                imageHR,
                noise_conf if noise_conf['add_noise_to_HR'] else 0
            )

        # Concatenate and save as a training image
        # if save_mode == 'test':
        #     save_image(config['dir'], config['date'], last_n+1+ni,
        #                imageLR_xz=imageLR, imageHR_xz=imageHR, mode='test')
        if psfconfigHR:
            # if ni in rperm:
            #     save_image(config['dir'], config['date'], last_n+1+ni,
            #                imageLR_xz=imageLR, imageHR_xz=imageHR, mode='val')
            # else:
            #     save_image(config['dir'], config['date'], last_n+1+ni,
            #                imageLR_xz=imageLR, imageHR_xz=imageHR, mode='train')
            save_image(config['dir'], None, last_n + 1 + ni,
                       imageLR_xz=imageLR, imageHR_xz=imageHR, mode='paired')
        else:
            save_image(config['dir'], None, last_n + 1 + ni,
                       imageLR_xz=imageLR, imageHR_xz=imageHR, mode='paired')
            # save_image(config['dir'], config['date'], last_n + 1 + ni,
            #            imageLR_xz=imageLR, imageHR_xz=imageHR, mode='process')

        # save_image(config['dir'], config['date'], last_n+1+ni,
        #             imageLR_xz=imageHR, imageHR_xz=None, mode='reference')

        print("Generated %i of %i images" % (ni + 1, n_images))
