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
    r'E:\LSM-deeplearning\TrainedModels\20210323_Speckle\20210323_Airy_1_BlastocystNS_l30_b1e-4'
]
in_dirs = [
    r'E:\LSM-deeplearning\20201008_BlastocystJC1_Processed\BlastocystJC1\150Radius-200exposure-0.5step-100mW-alpha1_1\Process-20201015-stack-p999pc'
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
