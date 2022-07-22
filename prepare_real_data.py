# -*- coding: utf-8 -*-
"""Crop ROIs from real images to use for training as the real salience contstraint
"""

# Contributions and Changelog accessed via a private git repository:
# https://github.com/philipwijesinghe/deep-learning-lsm.git
# philip.wijesinghe@gmail.com

# Copyright (c) 2022 Philip Wijesinghe@University of St Andrews (pw64@st-andrews.ac.uk)

from deeplearn.prepare_data import prepare_real_data

# for repeatability, set RNG seed
# np.random.seed(10)


# =============================================================================
# USER INPUT
# =============================================================================
# Path to a folder (or list of folders) that contains an image stack (image sequence) of your data
# -- Single
in_dirs = [r'E:\DataUnderpinning\ExampleAiry1\ExperimentalData']

# Path to the desired output folder (or list of folders) where to save the cropped images
out_dirs = [r'E:\DataUnderpinning\ExampleAiry1\RealTrainingData-example']


# Configuration : include in dict to modify
# config:
#     out_size = 64
#         size in pix of the output images
#     in_pix_size = 1
#         pixel size of input image
#     out_pix_size = 1
#         pixel size of output image
#     focus_centre = image_size_z // 2
#         index position in z where the focus is (ROIs will be selected from there)
#     crop_type = 'centre' | 'random' | 'brightest'
#         'centre' will select the central ROI of each image
#         'random' will select a random ROI for each image around the focus
#         'brightest' will select a random ROI around the brightest area in each image
#     normalise = True | False
#         normalise the output images to min max values
#     keep_percentile = 50
#         keeps only the images with intensity > percentile (0 will keep all)
config = {
    'crop_type': 'brightest',
}


# =============================================================================
# MAIN
# =============================================================================
for i, dir_ in enumerate(in_dirs):
    print('Selecting data from: %s' % dir_)
    prepare_real_data(dir_, out_dirs[i], **config)
