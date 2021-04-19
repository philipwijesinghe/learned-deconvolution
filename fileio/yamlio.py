# -*- coding: utf-8 -*-
"""yamlio

functions for reading and writing yaml configuration files
"""

# Created on Wed June  17  2020
#
# Contributions and Changelog accessed via a private git repository:
# https://github.com/philipwijesinghe/deep-learning-lsm.git
# philip.wijesinghe@gmail.com
#
# Copyright (c) 2020 Philip Wijesinghe@University of St Andrews (pw64@st-andrews.ac.uk)

import os
import yaml
import numpy as np

from lsm.psf import PSFConfig


# =============================================================================
# CONFIG CLASS
# =============================================================================
class TrainingConfig:
    """Training configuration pattern
    """

    def __init__(self):
        # Use a nested dict for now
        default_conf = {
            'date': 'default',
            'data': 'sim',
            'data_dir': '',
            'data_real_dir': '',
            'theta': -np.pi / 4,
            'img_size_xz': (64, 64),
            'pix_spacing': 0.85e-6,
            'n_val_real': 6,
            'sim': {
                'type': 'bead',
                'n_images': 8,
                'n_val': 2,
                'n_scatterers_min_max': (5, 40),
                'psfDistMaxLR': 24e-6,
                'psfDistMaxHR': 12e-6,
                'speckle': {
                    'hf_size_pix': 100,
                    'lf_size_pix': 5,
                    'weight': 0.001
                }
            },
            'noise': {
                'add_noise_to_HR': 0,
                'type': 'gaussian',
                'gaussian_mu': 0,
                'gaussian_std': 0.03,
                'saltpepper': 0.01,
                'noise_var': 0.2,
                'theta_var': 0.05,
            }
        }

        self.config = default_conf

    @staticmethod
    def _recursive_merge_strings(old, new):
        """Recursively merge the new dictionary into the config iterating
        through levels
        """
        for key, val in new.items():
            # Follow the default configuration template
            if key in old:
                # if nested
                if isinstance(old[key], dict):
                    # go again
                    TrainingConfig._recursive_merge_strings(old[key], new[key])
                # if string
                elif isinstance(old[key], str):
                    old[key] = str(new[key])
                # if a number
                elif isinstance(old[key], (int, float, complex)):
                    if isinstance(new[key], (int, float, complex)):
                        old[key] = new[key]
                    elif isinstance(new[key], str):
                        old[key] = eval(new[key])
                # if a tuple
                elif isinstance(old[key], tuple):
                    if isinstance(new[key], (tuple, list)):
                        old[key] = tuple(new[key])
            else:
                old[key] = new[key]

    def write_yaml(self, yaml_raw):
        # Recursively overwrite string values
        TrainingConfig._recursive_merge_strings(self.config, yaml_raw)

        # Special case for x Pi
        if 'theta' in yaml_raw:
            self.config['theta'] = eval(yaml_raw['theta']) * np.pi


# =============================================================================
# FUNCTIONS
# =============================================================================
def _read_yaml(yamlpath):
    # Reads yml into 'dict' object
    with open(yamlpath, 'r') as stream:
        try:
            config_raw = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return config_raw


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _psf_parser(psfconfig, config_raw, config_name='psfconfig'):
    # copy configuration keys
    if config_name in config_raw:
        if 'illumination' in config_raw[config_name]:
            conf_ = config_raw[config_name]['illumination']
            for key in conf_:
                psfconfig.config['illumination'][key] = \
                    float(conf_[key]) if is_number(conf_[key]) \
                    else conf_[key]

        if 'detection' in config_raw[config_name]:
            conf_ = config_raw[config_name]['detection']
            for key in conf_:
                psfconfig.config['detection'][key] = \
                    float(conf_[key]) if is_number(conf_[key]) \
                    else conf_[key]


def _parse_config(config_raw):
    # Low resolution PSF
    psfconfigLR = PSFConfig()
    _psf_parser(psfconfigLR, config_raw, config_name='psfconfigLR')

    # High resolution PSF
    psfconfigHR = PSFConfig()
    _psf_parser(psfconfigHR, config_raw, config_name='psfconfigHR')

    # Removing psfconfigs
    if 'psfconfigLR' in config_raw:
        del config_raw['psfconfigLR']
    if 'psfconfigHR' in config_raw:
        del config_raw['psfconfigHR']

    # Create config based on a class pattern
    trainconfig = TrainingConfig()
    trainconfig.write_yaml(config_raw)
    config = trainconfig.config

    return config, psfconfigLR, psfconfigHR


def load_config(yamlpath):
    config_raw = _read_yaml(yamlpath)

    config, psfconfigLR, psfconfigHR = _parse_config(config_raw)

    config['dir'] = os.path.dirname(os.path.abspath(yamlpath))

    return config, psfconfigLR, psfconfigHR


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Run this section as a script, while making all defined functions
    # available to the module

    # NOTE: uses parent modules
    # from beam import    will not work unless modules are loaded or the module
    # is run as a script with the '-m' modifier

    # load default configuration
    config, psfconfigLR, psfconfigHR = \
        load_config(r"F:\Work\Projects\deep-learning\deep-learning-lsm\config_templates\TrainConfig.yml")
