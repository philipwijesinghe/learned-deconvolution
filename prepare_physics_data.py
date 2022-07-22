# -*- coding: utf-8 -*-
"""Prepare data for training based on a specified config file
"""

# Created on Wed June 17 2020

# Contributions and Changelog accessed via a private git repository:
# https://github.com/philipwijesinghe/deep-learning-lsm.git
# philip.wijesinghe@gmail.com

# Copyright (c) 2020 Philip Wijesinghe@University of St Andrews (pw64@st-andrews.ac.uk)


from deeplearn.prepare_data import generate_physics_data

# for repeatability, set RNG seed
# np.random.seed(0)


# =============================================================================
# USER INPUT
# =============================================================================
# Path to a folder (or list of folders) that contains a PhysicsDataConfig.yml file
# -- Single
in_dirs = [r'E:\DataUnderpinning\ExampleAiry1\PhysicsData']
# -- List of folders
# in_dirs = [
#     r'E:\DataUnderpinning\PhysicsData\20210113_Airy_g1',
#     r'E:\DataUnderpinning\PhysicsData\20210222_Bessel_rr1p1_n5'
# ]
# -- All subfolders in a folder
# import glob
# in_dirs = glob.glob(r'E:\DataUnderpinning\PhysicsData\*')


# =============================================================================
# MAIN
# =============================================================================
for dir_ in in_dirs:
    print('Generating data in: %s' % dir_)
    generate_physics_data(dir_)
