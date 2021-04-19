# -*- coding: utf-8 -*-
"""Train a physics based model using unpaired real data
"""

# Created on 20201027

# Contributions and Changelog accessed via a private git repository:
# https://github.com/philipwijesinghe/deep-learning-lsm.git
# philip.wijesinghe@gmail.com

# Copyright (c) 2020 Philip Wijesinghe@University of St Andrews (pw64@st-andrews.ac.uk)

from deeplearn.prepare_data import prepare_training_data
from fileio.yamlio import load_config

import deeplearn.physics_based_training as dl
import yaml


# =============================================================================
# USER INPUT
# =============================================================================
networkdir = [
    r'E:\\LSM-deeplearning\\TrainedModels\\20210323_Speckle\\20210323_Airy_1_BlastocystNS_l30_b1e-4',
    r'E:\\LSM-deeplearning\\TrainedModels\\20210323_Speckle\\20210323_Gauss_100_BlastocystNS_l30_b1e-4'
]


# =============================================================================
# MAIN
# =============================================================================
for dir_ in networkdir:
    config, psfconfigLR, psfconfigHR = load_config(dir_ + '/TrainConfig.yml')

    # Prepare data for training
    prepare_training_data(dir_, overwrite=False)

    # Train
    conf = dl.Config()
    conf.overwrite_defaults(config)

    with open(dir_ + '//TrainRun.yml', 'w') as file:
        yaml.dump(conf, file)

    dl.train_model(conf, dir_)
