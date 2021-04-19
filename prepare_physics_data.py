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
# np.random.seed(10)


# =============================================================================
# USER INPUT
# =============================================================================
in_dirs = [r'E:\LSM-deeplearning\PhysicsData\20210323_Airy_g0p5_finer_speckle']


# =============================================================================
# MAIN
# =============================================================================
for dir_ in in_dirs:
    print('Generating data in: %s' % dir_)
    generate_physics_data(dir_)
