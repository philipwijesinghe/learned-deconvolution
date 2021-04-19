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

import deeplearn.simulation_training as dl
import yaml


# =============================================================================
# USER INPUT
# =============================================================================
# Folder with TrainConfig.yml
networkdir = [
    'E:\\LSM-deeplearning\\TrainedModels\\20210222_PureSimBeams\\20210222_Airy_g0p5_l30',
    'E:\\LSM-deeplearning\\TrainedModels\\20210222_PureSimBeams\\20210222_Airy_g1_l30',
    'E:\\LSM-deeplearning\\TrainedModels\\20210222_PureSimBeams\\20210222_Airy_g2_l30',
    'E:\\LSM-deeplearning\\TrainedModels\\20210222_PureSimBeams\\20210222_Bessel_rr0p8_n5',
    'E:\\LSM-deeplearning\\TrainedModels\\20210222_PureSimBeams\\20210222_Bessel_rr1p1_n5',
    'E:\\LSM-deeplearning\\TrainedModels\\20210222_PureSimBeams\\20210222_Gauss_100_l30',
    'E:\\LSM-deeplearning\\TrainedModels\\20210222_PureSimBeams\\20210222_Gauss_50_l30'
]


# =============================================================================
# MAIN
# =============================================================================
for dir_ in networkdir:
    config, psfconfigLR, psfconfigHR = load_config(dir_ + '/TrainConfig.yml')

    # Prepare data for training
    prepare_training_data(dir_, overwrite=False, data='sim')

    # Train
    conf = dl.Config()
    conf.overwrite_defaults(config)

    with open(dir_ + '//TrainRun.yml', 'w') as file:
        yaml.dump(conf, file)

    dl.train_model(conf, dir_)
