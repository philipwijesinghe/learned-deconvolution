# -*- coding: utf-8 -*-
""" Processes data with a trained model
"""

# Created on 20210128

# Contributions and Changelog accessed via a private git repository:
# https://github.com/philipwijesinghe/deep-learning-lsm.git
# philip.wijesinghe@gmail.com

# Copyright (c) 2021 Philip Wijesinghe@University of St Andrews (pw64@st-andrews.ac.uk)

import deeplearn.physics_based_training as dl
from deeplearn.inference import process_widefield_stack


# =============================================================================
# USER INPUT
# =============================================================================
model_dir = [
    r'G:\LSM-deeplearning\_DataUnderpinning\ExampleAiry1\TrainedModel'
]
in_dirs = [
    r'G:\LSM-deeplearning\_DataUnderpinning\ExampleAiry1\ExperimentalData'
]


# =============================================================================
# MAIN
# =============================================================================
config = dl.Config()
for i, dir_ in enumerate(in_dirs):
    print('Processing: %s' % dir_)
    if len(model_dir) == len(in_dirs):
        process_widefield_stack(dir_, model_dir[i], config)
    elif len(model_dir) == 1:
        process_widefield_stack(dir_, model_dir, config)
    else:
        print('Number of models and directories to process is mismatched')
