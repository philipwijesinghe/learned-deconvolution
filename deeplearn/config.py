# -*- coding: utf-8 -*-
""" Defines the Config Class for storing and setting DL parameters
"""

# Contributions and Changelog accessed via a private git repository:
# https://github.com/philipwijesinghe/deep-learning-lsm.git
# philip.wijesinghe@gmail.com

# Copyright (c) 2020 Philip Wijesinghe@University of St Andrews (pw64@st-andrews.ac.uk)


class Config:
    """Config Class for storing and modifying DL training parameters

    This Class stores a list of default training parameters as properties with a method for overwriting individual
    properties using a dictionary. This dictionary mirrors the parameters and format that is loaded form the
    user-specified TrainConfig.yml config file.

    """
    def __init__(self):
        """Config Class for storing and modifying DL training parameters

        Initialises default values

        """
        self.epoch = 0
        self.n_epochs = 301
        self.batch_size = 8
        self.lrG = 0.0001
        self.lrD = 0.0004
        self.b1 = 0.5
        self.b2 = 0.999
        self.lambda_pixel = 30
        self.beta_pixel = 0.0001
        self.gamma_pixel = 1
        self.decay_epoch = 100
        self.n_cpu = 0
        self.img_size = 64
        self.channels = 1
        self.sample_interval = 500
        self.checkpoint_interval = 100
        self.spectral_norm = False
        self.dual_D = True
        self.smooth_labels = True
        self.model = 'resnet'
        self.scheduler = False
        self.loss_decay = False
        # TODO: document idividual properties

    def overwrite_defaults(self, config):
        """Overwrites the Class properties with those specified in the config argument

        Properties overwritten are based on existing members of the list

        Parameters
        ----------
        config
            dictionary of configuration name : value pairs

        """
        self.n_epochs = config['n_epochs'] if 'n_epochs' in config else self.n_epochs
        self.batch_size = config['batch_size'] if 'batch_size' in config else self.batch_size
        self.b1 = config['b1'] if 'b1' in config else self.b1
        self.b2 = config['b2'] if 'b2' in config else self.b2
        self.model = config['model'] if 'model' in config else self.model
        self.lambda_pixel = config['lambda_pixel'] if 'lambda_pixel' in config else self.lambda_pixel
        self.beta_pixel = config['beta_pixel'] if 'beta_pixel' in config else self.beta_pixel
        self.gamma_pixel = config['gamma_pixel'] if 'gamma_pixel' in config else self.gamma_pixel
        self.smooth_labels = config['smooth_labels'] if 'smooth_labels' in config else self.smooth_labels
        self.spectral_norm = config['spectral_norm'] if 'spectral_norm' in config else self.spectral_norm
        self.dual_D = config['dual_D'] if 'dual_D' in config else self.dual_D
        self.lrD = config['lrD'] if 'lrD' in config else self.lrD
        self.lrG = config['lrG'] if 'lrG' in config else self.lrG
        self.scheduler = config['scheduler'] if 'scheduler' in config else self.scheduler
        self.loss_decay = config['loss_decay'] if 'loss_decay' in config else self.loss_decay
