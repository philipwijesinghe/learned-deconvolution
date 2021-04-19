# -*- coding: utf-8 -*-
""" Saves processed data (in and out) in few zx and xy slices
"""

# Created on 20210128

# Contributions and Changelog accessed via a private git repository:
# https://github.com/philipwijesinghe/deep-learning-lsm.git
# philip.wijesinghe@gmail.com

# Copyright (c) 2021 Philip Wijesinghe@University of St Andrews (pw64@st-andrews.ac.uk)

from fileio.visualise import visualise_net_in_out, Opts

in_dirs = [
    r'E:\LSM-deeplearning\20201007_BeadsRe_Processed\200nmBeads\135Radius_100esp_100mW_step0.5_alpha0.5_1'
]

for dir_ in in_dirs:
    opts = Opts()
    visualise_net_in_out(dir_, opts)
